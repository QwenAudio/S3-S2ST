import unittest
from tn_wer import clean, score_rows

class CorpusTests(unittest.TestCase):
    def test_zh_equivalence(self):
        for a,b in [('百分之十五','15%'),('百 分 之 十 五','15%'),('三点一四','3.14'),('负三点五','-3.5'),('三分之一','1/3'),('二零二四年五月一日','2024年5月1日'),('两个人','2个人'),('三美元','3美元'),('十点三十分','10:30'),('臺灣','台湾')]:
            with self.subTest(a=a): self.assertEqual(clean(a,'zh'),clean(b,'zh'))
    def test_preserve_semantics(self):
        for a,b in [('十五','百分之十五'),('三点一四','三十一点四'),('负三','三'),('啊哪儿','哪')]:
            self.assertNotEqual(clean(a,'zh'),clean(b,'zh'))
        self.assertEqual(clean('一方面一些东西','zh'),'一 方 面 一 些 东 西')
    def test_corpus(self):
        r=score_rows([('a','a','x'),('b','a b c d e f g h i','a b c d e f g h i')],'en')[2]
        self.assertEqual(r['score'],10)
    def test_empty(self):
        r=score_rows([('a','a b',''),('b','','x')],'en')[2]
        self.assertEqual(r['score'],150)
        self.assertEqual(r['insertions'],1)
    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError): score_rows([('a','x','x'),('a','x','x')],'en')
    def test_en(self):
        self.assertEqual(clean('fifteen percent','en'),clean('15%','en'))
        self.assertNotEqual(clean('3.14','en'),clean('314','en'))

if __name__=='__main__': unittest.main()
