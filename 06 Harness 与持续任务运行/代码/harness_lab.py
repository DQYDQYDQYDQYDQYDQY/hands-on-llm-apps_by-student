"""教学 Harness：真实只读工具、持续预算、检查点、暂停和不确定结果处理。"""
import asyncio,copy,hashlib,json,sys,time,uuid
from pathlib import Path
from checkpoint_store import CheckpointStore,encode
from runtime_events import EventLog
ROOT=Path(__file__).resolve().parents[2]
for directory in ['02 从大模型到 Agent 组成与最小循环', '04 上下文工程与记忆', '05 MCP 与 Skills 能力扩展']:
    sys.path.insert(0, str(ROOT / directory / '代码'))
from agent_lab import declaration,ModelError
from context_lab import atomic_message_groups,pack_context
from skill_lab import SkillLibrary
from skill_mcp_agent import SKILL_TOOLS,SYSTEM,mcp_declarations,dispatch
from mcp_connection import connect
try:
    from mcp import MCPError
except ImportError:  # mcp 2.x renamed the exported exception class.
    from mcp import McpError as MCPError

ARTIFACT_TOOL=declaration('read_artifact','读取当前任务保存的完整工具结果，ID 来自结果摘要。',
    {'id':{'type':'string'}},['id'])
ALLOWED={'activate_skill','read_skill_resource','mcp_list_files','mcp_read_file','read_artifact'}

def source_snapshot():
    paths=list((ROOT/'02 从大模型到 Agent 组成与最小循环/数据/demo-repo').glob('*'))
    paths+=list((ROOT/'05 MCP 与 Skills 能力扩展/代码/skills/repo-guide').rglob('*.md'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}

class LessonRuntime:
    """默认第六章环境；综合案例提供相同接口，复用状态机而不复制循环。"""
    identity='lesson06'
    allowed=ALLOWED
    def snapshot(self):return source_snapshot()
    def library(self):return SkillLibrary(ROOT/'05 MCP 与 Skills 能力扩展/代码/skills')
    def connection(self):return connect()
    def instructions(self,library):
        return SYSTEM+'\n可用 Skill 元数据：'+json.dumps(library.catalog(),ensure_ascii=False)
    def declarations(self,tools):return SKILL_TOOLS+mcp_declarations(tools)+[ARTIFACT_TOOL]
    async def execute(self,client,library,name,args,remote_names):
        return await dispatch(client,library,name,args,remote_names)

class SimulatedCrash(RuntimeError):
    """仅用于课堂中断注入，不是真正杀进程。跨进程恢复另由 CLI 验证。"""

class Harness:
    def __init__(self,store,run_id,log_path,runtime=None):
        self.store,self.run_id=store,run_id
        self.state,self.revision=store.load(run_id)
        self.runtime=runtime or LessonRuntime()
        if self.state.get('runtime_id','lesson06')!=self.runtime.identity:raise ValueError('任务运行环境不匹配')
        self.log=EventLog(log_path,self.state['trace_id'])

    @classmethod
    def create(cls,store,goal,log_path,max_turns=7,max_tools=12,context_chars=18000,runtime=None):
        if not isinstance(goal,str) or not goal.strip() or len(goal)>4000:raise ValueError('目标应为 1–4000 字符')
        if any(type(x) is not int or x<1 for x in [max_turns,max_tools,context_chars]):raise ValueError('预算必须为正整数')
        runtime=runtime or LessonRuntime()
        run_id=uuid.uuid4().hex
        state={'schema_version':1,'run_id':run_id,'trace_id':uuid.uuid4().hex,
            'goal':goal,'status':'active','phase':'model_ready','turns':0,'tool_calls':0,
            'limits':{'max_turns':max_turns,'max_tools':max_tools,'context_chars':context_chars},
            'messages':[],'pending':[],'next_tool':0,'seen_ids':[],
            'active_skills':[],'artifacts':[],'usage':[],'answer':'','source_snapshot':runtime.snapshot(),'runtime_id':runtime.identity}
        store.create(state)
        obj=cls(store,run_id,log_path,runtime=runtime)
        obj.log.emit('course.agent.started','started',span_id=state['trace_id'][:16])
        return obj

    def checkpoint(self):
        self.revision=self.store.save(self.state,self.revision)
        self.log.emit('course.checkpoint.completed','succeeded',metrics={'revision':self.revision})

    def stop(self,status):
        self.state['status']=status;self.checkpoint()
        self.log.emit('course.task.status_changed','succeeded',attributes={'task_status':status})
        return self.state

    def pause(self):
        if self.state['phase'] not in {'model_ready','tools_ready'}:raise ValueError('操作进行中不能当作干净暂停')
        if self.state['status']!='active':raise ValueError('当前任务未在运行')
        self.stop('paused')

    def resume(self):
        if self.state['status']!='paused':raise ValueError('只恢复明确暂停的任务')
        self.state['status']='active';self.checkpoint()

    def resolve_uncertain(self,decision):
        # 课堂仅允许显式重试已注册的只读/本地加载动作；不推广到外部写入。
        phase=self.state['phase']
        if phase not in {'model_inflight','tool_inflight'}:raise ValueError('没有待处理的不确定操作')
        if decision=='stop':return self.stop('cancelled')
        if decision!='retry':raise ValueError('决策只能是 retry 或 stop')
        if phase=='tool_inflight':
            name=self.state['pending'][self.state['next_tool']]['function']['name']
            if name not in self.runtime.allowed:raise PermissionError('不自动重试未知动作')
        self.state['phase']='model_ready' if phase=='model_inflight' else 'tools_ready'
        self.state['status']='active';self.checkpoint()
        # 已预留的请求/工具次数不退回；结果不确定也可能已消耗资源。
        self.log.emit('course.recovery.completed','succeeded',attributes={'decision':'explicit_retry','operation':phase})

    def build_context(self,library):
        instructions=self.runtime.instructions(library)
        # 已激活的方法跨进程重读；本课用固定自编包，资料升级需另建任务复核。
        for name in self.state['active_skills']:instructions+='\n已激活方法：\n'+library.activate(name)
        groups=atomic_message_groups(self.state['messages'])
        blocks=[{'label':f'history-{i}','priority':i,'messages':g} for i,g in enumerate(groups)]
        packed=pack_context(instructions,self.state['goal'],blocks,self.state['limits']['context_chars'])
        messages=packed.pop('messages');report=packed
        self.log.emit('course.context.completed','succeeded',
            metrics={'selected_groups':len(report['selected']),'dropped_groups':len(report['dropped']),
                     'serialized_chars':report['used_chars']})
        return messages,report

    async def step(self,model,crash_at=None):
        try:return await self._step(model,crash_at)
        except BaseExceptionGroup as group:
            # anyio 的资源退出会包装异常；单一原因为课堂提供可读异常。
            error=group
            while isinstance(error,BaseExceptionGroup) and len(error.exceptions)==1:error=error.exceptions[0]
            raise error from None

    async def _step(self,model,crash_at=None):
        if self.state['status']!='active':return self.state
        if self.state['phase'] in {'model_inflight','tool_inflight'}:
            return self.stop('needs_review')
        if self.state['source_snapshot']!=self.runtime.snapshot():return self.stop('source_changed')
        library=self.runtime.library()
        for name in self.state['active_skills']:library.activate(name)
        async with self.runtime.connection() as client:
            tools=(await client.list_tools()).tools
            remote_names={t.name for t in tools}
            declarations=self.runtime.declarations(tools)
            if self.state['phase']=='model_ready':
                if self.state['turns']>=self.state['limits']['max_turns']:return self.stop('turn_budget')
                try:messages,context_report=self.build_context(library)
                except ValueError:return self.stop('context_budget')
                self.state['last_context_report']=context_report
                self.state['last_model_input']=self.store.put_artifact(self.run_id,{'messages':messages,'tools':declarations})
                self.state['turns']+=1;self.state['phase']='model_inflight';self.checkpoint()
                span=self.log.emit('course.model.started','started',metrics={'request_ordinal':self.state['turns']})
                start=time.perf_counter();usage_start=len(getattr(model,'usage',[]))
                try:response=await asyncio.to_thread(model,messages,declarations)
                except ModelError:
                    self.log.emit('course.model.failed','failed',span,error={'category':'model','type':'ModelError','retryable':False})
                    return self.stop('needs_review')
                if crash_at=='after_model':raise SimulatedCrash('模型已返回，尚未保存响应')
                self.state['usage']+=getattr(model,'usage',[])[usage_start:]
                self.log.emit('course.model.completed','succeeded',span,metrics={'duration_ms':round((time.perf_counter()-start)*1000,3)})
                calls=response.get('tool_calls') or []
                if not isinstance(calls,list) or any(not isinstance(c,dict) or not isinstance(c.get('function'),dict)
                    or not isinstance(c['function'].get('name'),str) or not isinstance(c['function'].get('arguments'),str) for c in calls):
                    self.state['phase']='finished';return self.stop('invalid_model_output')
                if not calls:
                    self.state['answer']=response.get('content') or '';self.state['phase']='finished'
                    return self.stop('final_answer' if self.state['answer'].strip() else 'empty_answer')
                ids=[c.get('id') for c in calls]
                if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids) or set(ids).intersection(self.state['seen_ids']):
                    self.state['phase']='finished';return self.stop('invalid_call_ids')
                self.state['seen_ids']+=ids;self.state['messages'].append(response)
                self.state['pending']=calls;self.state['next_tool']=0;self.state['phase']='tools_ready';self.checkpoint()
                return self.state
            if self.state['phase']=='tools_ready':
                if self.state['tool_calls']>=self.state['limits']['max_tools']:return self.stop('tool_budget')
                call=self.state['pending'][self.state['next_tool']];fn=call.get('function',{})
                name=fn.get('name','');allowed=name in self.runtime.allowed
                self.log.emit('course.permission.completed','succeeded',attributes={'tool_name':name if allowed else '<unregistered>','decision':'allow' if allowed else 'deny'})
                self.state['tool_calls']+=1;self.state['phase']='tool_inflight';self.checkpoint()
                span=self.log.emit('course.tool.started','started',attributes={'tool_name':name if allowed else '<unregistered>'})
                start=time.perf_counter()
                try:
                    if not allowed:raise PermissionError('未授权工具')
                    args=json.loads(fn.get('arguments',''))
                    if name=='read_artifact':
                        if not isinstance(args,dict) or set(args)!={'id'}:raise ValueError('产物参数错误')
                        result={'ok':True,'data':self.store.read_artifact(self.run_id,args['id'])}
                    else:result=await self.runtime.execute(client,library,name,args,remote_names)
                except (ValueError,TypeError,KeyError,PermissionError,OSError,MCPError) as error:
                    result={'ok':False,'error_type':type(error).__name__,'message':'工具调用未完成'}
                if crash_at=='after_tool':raise SimulatedCrash('工具已返回，尚未保存结果')
                artifact=self.store.put_artifact(self.run_id,result)
                self.state['artifacts'].append({'call_id':call['id'],**artifact})
                # 短结果直接进上下文；长结果保留引用，让模型按需读取。
                visible=result if artifact['size_bytes']<=5000 or name=='read_artifact' else {
                    'ok':result.get('ok',False),'artifact_id':artifact['id'],'summary':'结果已保存，使用 read_artifact 读取完整内容'}
                self.state['messages'].append({'role':'tool','tool_call_id':call['id'],'content':encode(visible)})
                self.state['active_skills']=sorted(library.active)
                self.state['next_tool']+=1
                if self.state['next_tool']==len(self.state['pending']):
                    self.state['pending']=[];self.state['next_tool']=0;self.state['phase']='model_ready'
                else:self.state['phase']='tools_ready'
                self.checkpoint()
                self.log.emit('course.tool.completed' if result.get('ok') else 'course.tool.failed',
                    'succeeded' if result.get('ok') else 'failed',span,attributes={'tool_name':name if allowed else '<unregistered>'},
                    metrics={'duration_ms':round((time.perf_counter()-start)*1000,3)},payload=artifact)
                return self.state
        raise ValueError('未知执行阶段')

    async def drive(self,model,max_steps=40):
        if type(max_steps) is not int or max_steps<1:raise ValueError('步数必须为正整数')
        for _ in range(max_steps):
            if self.state['status']!='active':break
            await self.step(model)
        return copy.deepcopy(self.state)
