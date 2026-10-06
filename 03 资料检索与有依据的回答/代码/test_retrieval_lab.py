import tempfile
import unittest
from pathlib import Path
from retrieval_lab import *

class RetrievalTests(unittest.TestCase):
    def test_chunk_locations_match_source_and_no_useless_last_overlap(self):
        text = '甲\n乙\n丙\n丁\n戊'
        chunks = chunk_text(text, 'a.md', 3, 1)
        self.assertEqual([(c.start_line, c.end_line) for c in chunks], [(1,3),(3,5)])
        for c in chunks:
            self.assertEqual(c.text, '\n'.join(text.splitlines()[c.start_line-1:c.end_line]))
        self.assertNotEqual(chunks[0].id, chunk_text(text+'改', 'a.md',3,1)[0].id)

    def test_invalid_chunk_step(self):
        with self.assertRaises(ValueError): chunk_text('x','a',3,3)
        with self.assertRaises(ValueError): chunk_text('x','a',0,0)
        self.assertEqual(chunk_text('', 'a'), [])

    def test_tfidf_matches_independent_hand_calculation(self):
        chunks=[Chunk('a',1,1,'cat cat dog','a'),Chunk('b',1,1,'dog bird','b')]
        index=TfidfIndex(chunks)
        expected=(2*(math.log(3/2)+1))/math.sqrt((2*(math.log(3/2)+1))**2+1)
        self.assertAlmostEqual(index.search('cat')[0]['score'], expected)
        self.assertEqual(index.search('unknown'), [])
        self.assertEqual(cosine({}, {'x':1}), 0)

    def test_scope_skips_hidden_and_symlinks(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); (root/'a.md').write_text('可见', encoding='utf-8')
            (root/'.private.md').write_text('不可见', encoding='utf-8')
            try:
                (root/'link.md').symlink_to(root/'.private.md')
            except OSError as error:
                self.skipTest(f'当前 Windows 账户不能创建符号链接: {error}')
            self.assertEqual({c.source for c in load_corpus(root)}, {'a.md'})

    def test_empty_retrieval_never_calls_model(self):
        def forbidden(*args): raise AssertionError('不应请求模型')
        self.assertFalse(rag_answer('无结果',[],forbidden)['model_called'])

    def test_citation_presence_does_not_claim_semantic_correctness(self):
        evidence=[{'citation':'a.md:1-3','text':'timeout 为 30'}]
        result=check_citation_membership('timeout 为 999 [a.md:1-3]',evidence)
        self.assertEqual(result['unknown'], [])
        self.assertEqual(result['semantic_support'],'requires_claim_review')
        self.assertEqual(check_citation_membership('[b.md:1-2]',evidence)['unknown'], ['b.md:1-2'])

    def test_fusion_deduplicates_each_ranking(self):
        c=Chunk('a',1,1,'x','a'); h={'chunk':c,'score':1}
        self.assertAlmostEqual(reciprocal_rank_fusion([[h,h]])[0]['score'],1/61)

    def test_recall_uses_unique_ids(self):
        c=Chunk('a',1,1,'x','a'); h={'chunk':c,'score':1}
        self.assertEqual(recall_at_k([h,h],{c.id,'missing'}),.5)
        with self.assertRaises(ValueError): recall_at_k([],set())

if __name__=='__main__': unittest.main()
