import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from prepare_ham_scene_splits import regroup
from prepare_ham_texture_splits import check_components, verify_screen


def write_table(folder, name, fields, rows):
    path = folder/name
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TextureReleaseEvidence(unittest.TestCase):
    def test_independent_graph_preserves_singletons_and_role_precedence(self):
        rows = [{"image_id":i,"split_group":i,"split":role} for i,role in
                (("a","test"),("b","calibration"),("c","dev"),("d","train"))]
        pairs = [("a","b"),("b","c")]
        old = {r["image_id"]:r for r in rows}
        new = {r["image_id"]:r for r in regroup(rows,pairs)}
        check_components(old,new,pairs)
        self.assertEqual((new["a"]["split"],new["c"]["split_group"]),("dev","a"))
        new["d"]["split_group"]="a"
        with self.assertRaisesRegex(ValueError,"Independent component"):
            check_components(old,new,pairs)

    def test_requires_full_union_and_verifies_strong_and_loose_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            words,old,geometry,review=[Path(temp)/k for k in ("words","old","geometry","review")]
            for folder in (words,old,geometry,review):
                folder.mkdir()
            word_hash=write_table(words,"candidates.csv",["first","second"],[("a","b"),("c","d")])
            old_hash=write_table(old,"candidates.csv",["first","second"],[("a","b"),("e","f")])
            review_hash=write_table(review,"scores.csv",["first","second","source"],[("a","b","new_screen_hit")])
            def info(folder, **kwargs):
                (folder/"summary.json").write_text(json.dumps(kwargs), encoding="utf-8")
            info(words,status="complete",candidate_sha256=word_hash,old_candidate_sha256=old_hash)
            info(old,candidate_sha256=old_hash)
            expected={"candidate_pairs":3, "weak_edges_sha256":hashlib.sha256(b"a,b\nc,d\n").hexdigest(),
                      "strong_edges_sha256":hashlib.sha256(b"a,b\n").hexdigest()}
            rows=[("a","b",1,0,16,4),("c","d",1,0,8,2),("e","f",1,1,28,7)]
            def write_scores(values):
                h=write_table(geometry,"scores.csv",["first","second","in_pool","same_v14_group","inliers","cells"],values)
                info(geometry,max_rank=20,include_appearance=True,status="complete",scores_sha256=h,
                     word_candidates_sha256=word_hash,candidate_pairs=3)
                info(review,status="complete",scores_sha256=review_hash,geometry_sha256=h)
            write_scores(rows)
            self.assertEqual(verify_screen(words,old,geometry,review,expected),({("a","b"),("c","d")},{("a","b")}))
            write_scores(rows[:2])
            with self.assertRaisesRegex(ValueError,"no geometry score"):
                verify_screen(words,old,geometry,review,expected)
            write_scores([rows[0],rows[0],rows[2]])
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                verify_screen(words,old,geometry,review,expected)


if __name__=="__main__":
    unittest.main()
