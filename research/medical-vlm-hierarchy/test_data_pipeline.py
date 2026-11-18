import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from prepare_pad import conservative_rows,patient_splits
from audit_images import duplicate_candidates
from extract_features import read_inputs
from ham_reviewed_pairs import CONFIRMED, UNCERTAIN, reviewed_groups
from prepare_ham_scene_splits import regroup


class HamIdentityTests(unittest.TestCase):
    def test_scene_links_keep_analysis_units_and_prior_training_exposure(self):
        rows = [dict(image_id=k, analysis_group=k, split_group=k, split=role, label=k, representative="1")
                for k,role in zip("abcd", ["test","dev","train","calibration"])]
        actual = regroup(rows, [("a","b"),("b","c")])
        self.assertEqual([r["split"] for r in actual], ["train","train","train","calibration"])
        self.assertEqual([r["split_group"] for r in actual], ["a","a","a","d"])
        for old,new in zip(rows,actual):
            self.assertEqual([old[k] for k in ("analysis_group","label","representative")],
                             [new[k] for k in ("analysis_group","label","representative")])

    def test_transitive_review_and_uncertain_followup(self):
        rows = [{'lesion_id': k, 'image_id': k + '1', 'dx': 'nv'} for k in 'edcba']
        pairs = {CONFIRMED: [('c1', 'b1'), ('b1', 'a1')], UNCERTAIN: [('d1', 'c1')]}
        units, groups = reviewed_groups(rows, pairs)
        self.assertEqual(units, dict(a='a', b='a', c='a', d='d', e='e'))
        self.assertEqual(groups, dict(a='a', b='a', c='a', d='a', e='e'))

    def test_conflicting_confirmed_labels_fail_but_uncertain_labels_stay_separate(self):
        rows = [{'lesion_id': 'x', 'image_id': 'X', 'dx': 'nv'},
                {'lesion_id': 'y', 'image_id': 'Y', 'dx': 'mel'}]
        with self.assertRaises(ValueError):
            reviewed_groups(rows, {CONFIRMED: [('X','Y')], UNCERTAIN: []})
        units, groups = reviewed_groups(rows, {CONFIRMED: [], UNCERTAIN: [('X','Y')]})
        self.assertNotEqual(units['x'], units['y'])
        self.assertEqual(groups['x'], groups['y'])


class PatientIdentityTests(unittest.TestCase):
    def test_raw_lesion_id_collision_keeps_patients_together(self):
        rows=[{'patient_id':'p1','lesion_id':'1','diagnostic':'A'},
              {'patient_id':'p2','lesion_id':'1','diagnostic':'B'},
              {'patient_id':'p3','lesion_id':'3','diagnostic':'A'}]
        linked=conservative_rows(rows)
        self.assertEqual(linked[0]['patient_id'],linked[1]['patient_id'])
        self.assertNotEqual(linked[0]['lesion_id'],linked[1]['lesion_id'])
        splits=patient_splits(linked,17)
        self.assertEqual(splits[linked[0]['patient_id']],splits[linked[1]['patient_id']])

    def test_mixed_diagnoses_per_patient_valid_but_conflicting_pair_rejected(self):
        rows=[{'patient_id':'p1','lesion_id':'1','diagnostic':'A'},
              {'patient_id':'p1','lesion_id':'2','diagnostic':'B'}]
        self.assertEqual(len(patient_splits(rows,17)),1)
        rows.append({'patient_id':'p1','lesion_id':'1','diagnostic':'B'})
        with self.assertRaises(ValueError):patient_splits(rows,17)

    def test_near_hash_index_catches_three_bits_but_not_four(self):
        def row(i,h):return {'image_id':i,'dhash':f'{h:016x}','split':i,'sha256':i,'pixel_sha256':i}
        pairs=duplicate_candidates([row('a',0),row('b',7),row('c',15)])
        self.assertTrue(any(p['first']=='a' and p['second']=='b' for p in pairs))
        self.assertFalse(any(p['first']=='a' and p['second']=='c' for p in pairs))


class ImageAuditIntegrationTests(unittest.TestCase):
    def test_content_audit_and_tamper_guard(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=root/'images';images.mkdir();out=root/'audit'
            # Deliberate cross-split duplicate: must flag, never silently remove.
            Image.new('RGB',(16,16),(20,40,80)).save(images/'a.png')
            (images/'b.png').write_bytes((images/'a.png').read_bytes())
            manifest=root/'split.csv'
            manifest.write_text('image_id,label,split,representative\na.png,A,train,1\nb.png,A,test,1\n',encoding='utf-8')
            run=subprocess.run([sys.executable,str(Path(__file__).with_name('audit_images.py')),
                '--images',str(images),'--split',str(manifest),'--output',str(out)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            audit,rows=read_inputs(out)
            self.assertEqual(audit['status'],'review_required');self.assertEqual(audit['exact_cross_split_pairs'],1)
            self.assertEqual(len(rows),2)
            with (out/'manifest.csv').open('a',encoding='utf-8') as f:f.write('\n')
            with self.assertRaises(ValueError):read_inputs(out)

    def test_missing_images_block_feature_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'images').mkdir();out=root/'audit'
            split=root/'split.csv';split.write_text('image_id,label,split,representative\na,A,test,1\n',encoding='utf-8')
            r=subprocess.run([sys.executable,str(Path(__file__).with_name('audit_images.py')),
                '--images',str(root/'images'),'--split',str(split),'--output',str(out)],capture_output=True,text=True)
            self.assertEqual(r.returncode,2,r.stderr)
            with self.assertRaises(ValueError):read_inputs(out)


if __name__=='__main__':unittest.main(verbosity=2)
