"""恢复、预算、产物与不确定操作的行为测试；真实 MCP，模型使用回放。"""
import asyncio,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from checkpoint_store import CheckpointStore,CheckpointConflict
from harness_lab import Harness,SimulatedCrash,ROOT
from runtime_events import EventLog
from agent_lab import ReplayModel,tool_response

class StoreTests(unittest.TestCase):
    def test_conflict_and_artifact_scope(self):
        with tempfile.TemporaryDirectory() as d:
            db=CheckpointStore(Path(d)/'run.db')
            db.create({'run_id':'one'});db.create({'run_id':'two'})
            state,revision=db.load('one');db.save(state,revision)
            with self.assertRaises(CheckpointConflict):db.save(state,revision)
            a=db.put_artifact('one',{'text':'原文'})
            self.assertEqual(db.read_artifact('one',a['id']),{'text':'原文'})
            with self.assertRaises(KeyError):db.read_artifact('two',a['id'])
            db.close()

    def test_logging_failure_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            blocked=Path(d)/'parent';blocked.write_text('file', encoding='utf-8')
            log=EventLog(blocked/'events.jsonl','1'*32)
            log.emit('course.test.started','started')
            self.assertEqual(log.dropped,1)

    def test_fresh_process_resumes_without_repeating_saved_tools(self):
        base=ROOT/'.local/student-runs/l06-tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as d:
            path=Path(d)/'run.db'
            command=[sys.executable, '-X', 'utf8',str(Path(__file__).with_name('harness_cli.py')),'--db',str(path)]
            first=json.loads(subprocess.check_output(command+['--goal','脚本恢复实验','--steps','3'],text=True, encoding='utf-8'))
            self.assertEqual(first['state']['tool_calls'],1)
            second=json.loads(subprocess.check_output(command+['--run-id',first['run_id']],text=True, encoding='utf-8'))
            self.assertEqual(second['state']['status'],'final_answer')
            self.assertEqual(second['state']['tool_calls'],5)
            ids=[a['call_id'] for a in second['state']['artifacts']]
            self.assertEqual(len(ids),len(set(ids)))

class HarnessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.store=CheckpointStore(self.root/'run.db')
    async def asyncTearDown(self):self.store.close();self.temp.cleanup()
    def harness(self,**kwargs):return Harness.create(self.store,'导读配置',self.root/'events.jsonl',**kwargs)

    async def test_clean_checkpoint_resume_and_context_rebuild(self):
        h=self.harness();model=ReplayModel([tool_response('a','activate_skill',{'name':'repo-guide'}),
            tool_response('b','mcp_read_file',{'path':'config.py'}),{'role':'assistant','content':'完成'}])
        await h.step(model);await h.step(model);h.pause()
        restored=Harness(self.store,h.run_id,self.root/'events.jsonl');restored.resume()
        state=await restored.drive(model)
        self.assertEqual(state['status'],'final_answer');self.assertEqual(state['tool_calls'],2)
        self.assertEqual(state['active_skills'],['repo-guide'])
        request=self.store.read_artifact(h.run_id,state['last_model_input']['id'])
        self.assertIn('已激活方法',request['messages'][0]['content'])

    async def test_uncertain_tool_requires_explicit_resolution(self):
        h=self.harness();model=ReplayModel([tool_response('a','mcp_read_file',{'path':'config.py'}),{'role':'assistant','content':'完成'}])
        await h.step(model)
        with self.assertRaises(SimulatedCrash):await h.step(model,crash_at='after_tool')
        restored=Harness(self.store,h.run_id,self.root/'events.jsonl')
        await restored.step(model);self.assertEqual(restored.state['status'],'needs_review')
        self.assertEqual(restored.state['tool_calls'],1)
        restored.resolve_uncertain('retry');state=await restored.drive(model)
        self.assertEqual(state['status'],'final_answer');self.assertEqual(state['tool_calls'],2)
        self.assertEqual(len(state['artifacts']),1)

    async def test_model_uncertainty_keeps_reserved_budget(self):
        h=self.harness(max_turns=1)
        with self.assertRaises(SimulatedCrash):
            await h.step(ReplayModel([{'role':'assistant','content':'返回但未保存'}]),crash_at='after_model')
        restored=Harness(self.store,h.run_id,self.root/'events.jsonl');restored.resolve_uncertain('retry')
        state=await restored.drive(ReplayModel([]))
        self.assertEqual(state['status'],'turn_budget');self.assertEqual(state['turns'],1)

    async def test_permission_and_tool_budget_persist(self):
        h=self.harness(max_tools=1)
        m=ReplayModel([tool_response('a','write_file',{'path':'x'}),tool_response('b','mcp_list_files',{})])
        state=await h.drive(m)
        self.assertEqual(state['status'],'tool_budget');self.assertEqual(state['tool_calls'],1)
        result=self.store.read_artifact(h.run_id,state['artifacts'][0]['id'])
        self.assertEqual(result['error_type'],'PermissionError')

    async def test_context_limit_and_source_change_stop_before_model(self):
        h=self.harness(context_chars=1)
        state=await h.drive(ReplayModel([]));self.assertEqual(state['status'],'context_budget')
        self.assertEqual(state['turns'],0)
        h=self.harness()
        with patch('harness_lab.source_snapshot',return_value={'changed':'version'}):
            state=await h.drive(ReplayModel([]))
        self.assertEqual(state['status'],'source_changed');self.assertEqual(state['turns'],0)

if __name__=='__main__':unittest.main()
