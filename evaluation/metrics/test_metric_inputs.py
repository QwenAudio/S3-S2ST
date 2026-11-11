import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from metric_inputs import read_keyed, read_asr_tsv, full_keys

class FullCoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def write(self,name,text):
        p=self.root/name;p.write_text(text);return p
    def test_empty_keyed_and_asr_are_retained(self):
        ref=read_keyed(self.write('label','a\tone two three four\nb\t\n'))
        _,hyp=read_asr_tsv(self.write('raw','/a.wav\t0\tone two three four\tone two three four\n/b.wav\t0\t\t\n'))
        self.assertEqual(full_keys(ref,hyp),['a','b'])
        self.assertEqual(hyp['b'],'')
    def test_duplicate_and_missing_are_rejected(self):
        with self.assertRaises(ValueError):read_keyed(self.write('dup','a\tone\na\ttwo\n'))
        with self.assertRaises(ValueError):read_asr_tsv(self.write('dup_asr','/a.wav\t0\ta\tb\n/a.wav\t0\ta\tc\n'))
        with self.assertRaises(ValueError):full_keys({'a':'a','b':'b'},{'a':'a'})
        with self.assertRaises(ValueError):full_keys({'a':'a'},{'a':'a','b':'b'})
    def test_malformed_asr_rejected(self):
        with self.assertRaises(ValueError):read_asr_tsv(self.write('bad','a\t0\ttext\n'))
    def test_comet_cli_keeps_empty_hypothesis(self):
        self.write('source','a\tsource a\nb\tsource b\n')
        self.write('label','a\treference a\nb\treference b\n')
        raw=self.write('raw','/a.wav\t0\ta\t你 好\n/b.wav\t0\tb\t\n')
        out=self.root/'comet.jsonl'
        subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_asr_comet.py')),'--recog-dir',str(self.root),'--asr-tsv',str(raw),'--target-language','zh','--output',str(out)],check=True,capture_output=True)
        rows=[json.loads(x) for x in out.read_text().splitlines()]
        self.assertEqual([x['mt'] for x in rows],['你好',''])
    def test_bleu_cli_rejects_partial_input(self):
        ref=self.write('label','a\tone two three four\nb\tfive six seven eight\n')
        raw=self.write('raw','/a.wav\t0\tone two three four\tone two three four\n')
        p=subprocess.run([sys.executable,str(Path(__file__).with_name('score_bleu.py')),'--ref',str(ref),'--asr-tsv',str(raw),'--language','en','--output',str(self.root/'bleu.json')],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('coverage mismatch',p.stderr)

if __name__=='__main__':unittest.main()
