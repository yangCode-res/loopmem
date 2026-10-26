"""Fresh offline human-reward Controller RL; fixed BGE and unchanged runtime schema.

Exact two-action contextual-bandit PPO: enumerate CONTINUE/SWITCH_LOOP,
refresh behavior probabilities each epoch, retain initial policy as KL reference.
This is not a rollout-level counterfactual estimator or online GRPO.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Dataset

ACTIONS = ('CONTINUE', 'SWITCH_LOOP')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def read_data(directory):
    m=json.loads((directory/'manifest.json').read_text())
    if m.get('training_ready') is not True or m['safety']!={
        'train_source_verified':True,'test_question_overlap':0,'qid_split_overlap':0,
        'actual_causal_inputs_unchanged':True,'no_external_annotation_api':True}:
        raise ValueError('data safety manifest invalid')
    parts=[]
    for name in ['train.jsonl','validation.jsonl']:
        if sha(directory/name)!=m['files'][name]:
            raise ValueError('data hash mismatch')
        rows=[json.loads(l) for l in (directory/name).read_text().splitlines()]
        if not rows or set(r['target_action'] for r in rows)!=set(ACTIONS):
            raise ValueError('both actions required in each partition')
        if len({r['sample_id'] for r in rows})!=len(rows):
            raise ValueError('duplicate event')
        parts.append(rows)
    if {r['qid'] for r in parts[0]} & {r['qid'] for r in parts[1]}:
        raise ValueError('qid leakage')
    if abs(sum(r['weight'] for r in parts[0])/len(parts[0])-1)>1e-5:
        raise ValueError('weights must be globally mean-normalized')
    return *parts,m


class ActionData(Dataset):
    def __init__(self,rows,tok,max_length):
        self.rows=rows
        self.items=[]
        prefix='{"decision": "'
        prefix_ids=tok.encode(prefix,add_special_tokens=False)
        for r in rows:
            if [m['role'] for m in r['messages']]!=['system','user']:
                raise ValueError('noncausal messages')
            prompt=tok.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True)
            candidates=[]
            for action in ACTIONS:
                target=tok.encode(prefix+action+'"',add_special_tokens=False)
                n=0
                while n<min(len(prefix_ids),len(target)) and prefix_ids[n]==target[n]:
                    n+=1
                if n==len(target):
                    raise ValueError('empty action span')
                ids=prompt+target
                if len(ids)>max_length:
                    raise ValueError(f'No silent truncation: {r["sample_id"]} has {len(ids)} tokens')
                candidates.append((ids,[-100]*(len(prompt)+n)+target[n:]))
            if r['weight']<=0 or not all(math.isfinite(x) for x in r['reward']):
                raise ValueError('invalid weight/reward')
            self.items.append(candidates)
        lengths=[len(c[0]) for item in self.items for c in item]
        self.audit={'rows':len(rows),'max_tokens':max(lengths),'mean_tokens':sum(lengths)/len(lengths),
                    'truncated_rows':0}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        return i,self.items[i],self.rows[i]


def collate(items,pad):
    candidates=[c for _,pair,_ in items for c in pair]
    maximum=max(len(c[0]) for c in candidates)
    ids=[];labels=[];mask=[]
    for x,y in candidates:
        n=maximum-len(x)
        ids.append(x+[pad]*n);labels.append(y+[-100]*n);mask.append([1]*len(x)+[0]*n)
    return {'input_ids':torch.tensor(ids),'attention_mask':torch.tensor(mask),'labels':torch.tensor(labels),
            'indices':torch.tensor([i for i,_,_ in items]),
            'rewards':torch.tensor([r['reward'] for _,_,r in items],dtype=torch.float32),
            'weights':torch.tensor([r['weight'] for _,_,r in items],dtype=torch.float32)}


def scores(model,batch,device):
    out=model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device))
    labels=batch['labels'].to(device)[:,1:]
    positions=labels.ne(-100).nonzero(as_tuple=False)
    # Softmax only action positions, not the entire context x vocabulary in fp32.
    logits=out.logits[positions[:,0],positions[:,1],:].float()
    targets=labels[positions[:,0],positions[:,1]]
    token_lp=logits.log_softmax(-1).gather(1,targets[:,None]).squeeze(1)
    sequence=torch.zeros(labels.shape[0],device=device).index_add(0,positions[:,0],token_lp)
    return sequence.reshape(-1,2)


def ppo_loss(new_scores,behavior,reference,rewards,weights,clip=.2,kl_coef=.05):
    logp=new_scores.log_softmax(-1)
    old=behavior.exp()
    advantage=rewards-(old*rewards).sum(-1,keepdim=True)
    ratio=(logp-behavior).clamp(-20,20).exp()
    surrogate=torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage)
    policy=-(old*surrogate).sum(-1)
    kl=(logp.exp()*(logp-reference)).sum(-1)
    # DO NOT divide by local weights.sum(): batch=1 would cancel every weight.
    loss=(weights*(policy+kl_coef*kl)).mean()
    return loss,{'policy_loss':float((weights*policy).mean().detach()),
                 'reference_kl':float(kl.mean().detach()),
                 'clip_fraction':float(((ratio-1).abs()>clip).float().mean().detach())}


@torch.no_grad()
def collect(model,data,loader,device):
    model.eval()
    result=torch.empty(len(data),2)
    for b in loader:
        result[b['indices']]=scores(model,b,device).log_softmax(-1).cpu()
    return result


def metrics(gold,pred):
    # Pred -1 represents invalid generation and remains an error in all denominators.
    cm=[[sum(g==i and p==j for g,p in zip(gold,pred)) for j in [0,1,-1]] for i in [0,1]]
    recall=[cm[i][i]/max(1,sum(cm[i])) for i in [0,1]]
    precision=cm[1][1]/max(1,cm[0][1]+cm[1][1])
    return {'n':len(gold),'accuracy':sum(g==p for g,p in zip(gold,pred))/len(gold),
            'switch_recall':recall[1],'continue_recall':recall[0],
            'switch_precision':precision,'switch_f1':2*precision*recall[1]/max(1e-12,precision+recall[1]),
            'balanced_accuracy':sum(recall)/2,'confusion_rows_C_S_columns_C_S_invalid':cm}


@torch.no_grad()
def generate_eval(model,tok,rows,device,path):
    model.eval()
    details=[]
    for r in rows:
        ids=tok.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(device)
        result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                              max_new_tokens=96,pad_token_id=tok.pad_token_id,use_cache=True)
        text=tok.decode(result[0,ids.shape[1]:],skip_special_tokens=True).strip()
        pred=-1;schema=False;key_ok=False
        try:
            obj=json.loads(text)
            schema=isinstance(obj,dict) and set(obj)=={'decision','closed_loop_key'} and obj['decision'] in ACTIONS and isinstance(obj['closed_loop_key'],str)
            if schema:
                pred=ACTIONS.index(obj['decision'])
                active=json.loads(r['messages'][1]['content'])['active_loop']['loop_key']
                key_ok=obj['closed_loop_key']==(active if pred==1 else '')
        except (ValueError,TypeError,KeyError):
            pass
        details.append({'sample_id':r['sample_id'],'qid':r['qid'],'target':r['target_action'],
                        'text':text,'prediction_index':pred,'schema_valid':schema,'closed_key_valid':key_ok})
        if len(details)%10==0:
            print(json.dumps({'generation_eval':path.name,'done':len(details),'total':len(rows)}),flush=True)
    dump(path,details)
    m=metrics([ACTIONS.index(r['target_action']) for r in rows],[d['prediction_index'] for d in details])
    m.update(schema_valid_rate=sum(d['schema_valid'] for d in details)/len(rows),
             closed_key_valid_rate=sum(d['closed_key_valid'] for d in details)/len(rows),
             exact_contract_accuracy=sum(d['closed_key_valid'] and ACTIONS.index(d['target'])==d['prediction_index'] for d in details)/len(rows))
    return m


def main():
    from functools import partial
    from peft import PeftModel
    from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
    p=argparse.ArgumentParser()
    p.add_argument('--base-model',required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--lr',type=float,default=5e-6)
    p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--max-length',type=int,default=8192)
    p.add_argument('--seed',type=int,default=20260906)
    a=p.parse_args()
    if a.epochs<1 or a.accumulation<1 or a.lr<=0:
        raise ValueError('invalid optimization settings')
    if not torch.cuda.is_available():
        raise ValueError('GPU required for actual RL training')
    train_rows,dev_rows,manifest=read_data(a.data)
    initial_files={str(f.name):sha(f) for f in a.adapter.glob('adapter*') if f.is_file()}
    if 'adapter_model.safetensors' not in initial_files:
        raise ValueError('initial adapter incomplete')
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.time();set_seed(a.seed)
    tok=AutoTokenizer.from_pretrained(a.base_model,local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    train=ActionData(train_rows,tok,a.max_length);dev=ActionData(dev_rows,tok,a.max_length)
    make_batch=partial(collate,pad=tok.pad_token_id)
    tloader=DataLoader(train,batch_size=1,collate_fn=make_batch)
    dloader=DataLoader(dev,batch_size=1,collate_fn=make_batch)
    dump(a.output/'input_audit.json',{'train':train.audit,'validation':dev.audit,'data_manifest_sha256':sha(a.data/'manifest.json'),
                                    'initial_adapter_hashes':initial_files,'algorithm':'offline_enumerated_action_PPO','arguments':vars(a)|{'adapter':str(a.adapter),'data':str(a.data),'output':str(a.output)}})
    base=AutoModelForCausalLM.from_pretrained(a.base_model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,str(a.adapter),is_trainable=True).to('cuda')
    model.config.use_cache=False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):
            module.p=0.0
    device=torch.device('cuda')
    reference=collect(model,train,tloader,device)
    initial_scored=collect(model,dev,dloader,device)
    gold=[ACTIONS.index(r['target_action']) for r in dev_rows]
    initial={'forced_two_action':metrics(gold,initial_scored.argmax(-1).tolist()),
             'greedy_json':generate_eval(model,tok,dev_rows,device,a.output/'initial_generations.json')}
    dump(a.output/'initial_validation.json',initial)
    print(json.dumps({'initial_validation':initial}),flush=True)
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=a.lr,weight_decay=0.0)
    history=[];steps=0
    for epoch in range(a.epochs):
        behavior=reference if epoch==0 else collect(model,train,tloader,device)
        loader=DataLoader(train,batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(a.seed+epoch),collate_fn=make_batch)
        model.train();optimizer.zero_grad(set_to_none=True)
        for i,b in enumerate(loader):
            loss,info=ppo_loss(scores(model,b,device),behavior[b['indices']].to(device),reference[b['indices']].to(device),
                               b['rewards'].to(device),b['weights'].to(device))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            # Correct denominator for the final, partially filled accumulation window.
            window=min(a.accumulation,len(loader)-(i//a.accumulation)*a.accumulation)
            (loss/window).backward()
            if (i+1)%a.accumulation==0 or i+1==len(loader):
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
                if not torch.isfinite(grad):
                    raise ValueError('nonfinite gradients')
                optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
            if (i+1)%10==0 or i+1==len(loader):
                print(json.dumps({'epoch':epoch+1,'batch':i+1,'total_batches':len(loader),'loss':float(loss.detach()),'optimizer_steps':steps,**info}),flush=True)
        checkpoint=a.output/f'epoch-{epoch+1}'
        model.save_pretrained(checkpoint,safe_serialization=True)
        scored=collect(model,dev,dloader,device)
        result=metrics(gold,scored.argmax(-1).tolist())
        history.append({'epoch':epoch+1,'forced_two_action':result})
        dump(a.output/'epoch_metrics.json',history)
        torch.save({'optimizer':optimizer.state_dict(),'epoch':epoch+1,'steps':steps},checkpoint/'optimizer.pt')
    final=a.output/'final_adapter'
    model.save_pretrained(final,safe_serialization=True);tok.save_pretrained(final)
    final_eval={'forced_two_action':history[-1]['forced_two_action'],
                'greedy_json':generate_eval(model,tok,dev_rows,device,a.output/'final_generations.json')}
    if {str(f.name):sha(f) for f in a.adapter.glob('adapter*') if f.is_file()}!=initial_files:
        raise ValueError('original adapter was modified')
    final_hash=sha(final/'adapter_model.safetensors')
    if final_hash==initial_files['adapter_model.safetensors']:
        raise ValueError('weights did not change')
    report={'completed':True,'algorithm':'offline_enumerated_action_PPO','initial_adapter':str(a.adapter),
            'data_manifest_sha256':sha(a.data/'manifest.json'),'initial_validation':initial,'final_validation':final_eval,
            'epochs':a.epochs,'optimizer_steps':steps,'elapsed_seconds':time.time()-started,
            'final_adapter_sha256':final_hash,'initial_adapter_unchanged':True,'reranker_trained':False,
            'boundary_reward_is_human_judgment_not_causal_counterfactual':True,
            'end_to_end_answer_improvement_tested':False}
    dump(a.output/'completion.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
