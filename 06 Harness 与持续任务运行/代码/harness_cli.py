"""每次启动都是新进程；从同一 SQLite 检查点继续课堂任务。"""
import argparse,asyncio,json
from pathlib import Path
from harness_lab import Harness,CheckpointStore,ROOT
from agent_lab import DeepSeekModel,tool_response,validate_deepseek_model

def demo_model(messages,tools):
    """固定的教学动作表，以已保存消息确定游标；不是模型自主选择。"""
    actions=[('activate_skill',{'name':'repo-guide'}),
             ('read_skill_resource',{'name':'repo-guide','path':'references/conditions.md'}),
             ('mcp_list_files',{}),('mcp_read_file',{'path':'main.py'}),
             ('mcp_read_file',{'path':'config.py'})]
    done=sum(len(m.get('tool_calls',[])) for m in messages)
    if done<len(actions):return tool_response(f'demo-{done}',*actions[done])
    return {'role':'assistant','content':'教学脚本已读完方法与源码；这是恢复机制演示，不能证明模型讲解正确。'}

async def main():
    p=argparse.ArgumentParser();p.add_argument('--db',required=True);p.add_argument('--run-id')
    p.add_argument('--goal');p.add_argument('--mode',choices=['replay','deepseek','inspect'],default='replay')
    p.add_argument('--steps',type=int,default=40);p.add_argument('--resolve',choices=['retry','stop']);args=p.parse_args()
    path=(ROOT/args.db).resolve();path.relative_to((ROOT/'.local/student-runs').resolve())
    store=CheckpointStore(path)
    try:
        if args.run_id:h=Harness(store,args.run_id,path.with_suffix('.jsonl'))
        else:
            if not args.goal:raise ValueError('新任务需要 --goal')
            h=Harness.create(store,args.goal,path.with_suffix('.jsonl'))
        if args.resolve:h.resolve_uncertain(args.resolve)
        if args.mode!='inspect':
            if args.mode=='deepseek':
                selected,available=validate_deepseek_model()
                print('本次选择的模型:',selected)
                print('当前账号可用模型:',available)
                model=DeepSeekModel(selected)
            else:model=demo_model
            await h.drive(model,max_steps=args.steps)
        print(json.dumps({'run_id':h.run_id,'state':h.state,'revision':h.revision},ensure_ascii=False))
    finally:store.close()

if __name__=='__main__':asyncio.run(main())
