import copy,json,tempfile,unittest
from pathlib import Path
from context_lab import *

class ContextTests(unittest.TestCase):
    def calls(self):
        return [{'role':'assistant','content':None,'tool_calls':[{'id':'a'},{'id':'b'}]},
                {'role':'tool','tool_call_id':'b','content':'B'},
                {'role':'tool','tool_call_id':'a','content':'A'}]
    def test_tools_are_atomic_even_when_results_reordered(self):
        messages=self.calls();self.assertEqual(atomic_message_groups(messages),[messages])
        with self.assertRaises(ValueError):atomic_message_groups(messages[1:])
        with self.assertRaises(ValueError):atomic_message_groups(messages[:-1])
    def test_duplicate_and_mismatched_ids_fail(self):
        messages=self.calls();messages[1]['tool_call_id']='a'
        with self.assertRaises(ValueError):atomic_message_groups(messages)
        with self.assertRaises(ValueError):atomic_message_groups(self.calls()+self.calls())
    def test_priority_selection_preserves_order_and_pinned_messages(self):
        blocks=[{'label':'old-important','priority':10,'messages':[{'role':'user','content':'旧的关键条件'}]},
                {'label':'noise','priority':0,'messages':[{'role':'user','content':'无关'*200}]},
                {'label':'new-important','priority':9,'messages':[{'role':'assistant','content':'新的结果'}]}]
        original=copy.deepcopy(blocks)
        result=pack_context('固定约束','当前目标',blocks,200)
        self.assertEqual(result['selected'],['old-important','new-important'])
        self.assertEqual(result['dropped'],['noise']);self.assertLessEqual(result['used_chars'],200)
        self.assertEqual(result['messages'][0]['content'],'固定约束')
        self.assertEqual(result['messages'][-1]['content'],'当前目标')
        self.assertEqual(blocks,original)
    def test_never_partially_keeps_tool_group(self):
        blocks=[{'label':'tools','priority':1,'messages':self.calls()}]
        r=pack_context('s','q',blocks,json_chars([{'role':'system','content':'s'},{'role':'user','content':'q'}])+5)
        self.assertEqual(r['selected'],[])
        with self.assertRaises(ValueError):pack_context('x'*100,'q',[],10)
    def test_store_isolated_persistent_stale_and_forget(self):
        memory={'goal':'学 Loss','pending_question':'p=0.25','hints_only':True,'source_version':'v1'}
        with tempfile.TemporaryDirectory() as name:
            a=MemoryStore(name);a.save('student-a',memory)
            b=MemoryStore(name)
            self.assertEqual(b.load('student-a','v1')['record'],memory)
            self.assertEqual(b.load('student-b','v1')['status'],'missing')
            self.assertEqual(b.load('student-a','v2')['status'],'stale')
            b.forget('student-a');self.assertEqual(a.load('student-a','v1')['status'],'missing')
            self.assertFalse(list(Path(name).glob('*.tmp')))
    def test_path_and_schema_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            store=MemoryStore(name)
            with self.assertRaises(ValueError):store.load('../outside','v1')
            with self.assertRaises(ValueError):store.save('a',{'goal':'不完整'})
            try:
                Path(name,'a.json').symlink_to(Path(name,'b.json'))
            except OSError as error:
                if getattr(error, 'winerror', None) == 1314:
                    self.skipTest('Windows 当前账户没有创建符号链接的权限')
                raise
            with self.assertRaises(ValueError):store.load('a','v1')
    def test_variants_hold_system_and_current_question_constant(self):
        p=Path(__file__).resolve().parents[1]/'数据/learning-session.json'
        fixture=json.loads(p.read_text(encoding='utf-8'));variants=context_variants(fixture)
        self.assertEqual(len({v[0]['content'] for v in variants.values()}),1)
        self.assertEqual(len({v[-1]['content'] for v in variants.values()}),1)
        self.assertNotIn('0.25',json.dumps(variants['recent_only']))
        self.assertIn('0.25',json.dumps(variants['curated_memory']))

if __name__=='__main__':unittest.main()
