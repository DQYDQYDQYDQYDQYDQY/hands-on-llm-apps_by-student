"""课程结构化事件：仅记录元信息，不把提示、源码或密钥写进日志。"""
import json,uuid
from datetime import datetime,timezone
from pathlib import Path

class EventLog:
    def __init__(self,path,trace_id):
        self.path,self.trace_id=Path(path),trace_id
        self.dropped=0

    def emit(self,event_name,status,span_id=None,attributes=None,metrics=None,payload=None,error=None):
        span_id=span_id or uuid.uuid4().hex[:16]
        event={'schemaVersion':'1','timestamp':datetime.now(timezone.utc).isoformat(),
            'eventName':event_name,'stage':event_name.rsplit('.',1)[-1],
            'status':status,'severity':'error' if status=='failed' else 'info',
            'trace':{'traceId':self.trace_id,'spanId':span_id,'parentSpanId':self.trace_id[:16],
                     'traceparent':f'00-{self.trace_id}-{span_id}-01'},
            'attributes':attributes or {},'metrics':metrics or {}}
        if span_id==self.trace_id[:16]:event['trace'].pop('parentSpanId')
        if payload is not None:event['payload']={'contentMode':'metadata_only',**payload}
        if error is not None:event['error']=error
        # 诊断写入失败不改变任务结果；检查点失败则必须阻止继续执行。
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
        except (OSError,ValueError,TypeError):self.dropped+=1
        return span_id
