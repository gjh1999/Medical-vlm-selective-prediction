import math
import unittest
from selective_core import Taxonomy,upper_bound,conformal_quantile,lac_set,select_policy
from prepare_ham import grouped_splits


class ResearchContractTests(unittest.TestCase):
    def setUp(self):self.t=Taxonomy({'a':'g','b':'g','c':'root','g':'root'})
    def test_root_is_not_free_correctness(self):
        r=self.t.metrics(['a','b'],['root','root'])
        self.assertEqual(r['node_coverage'],0);self.assertIsNone(r['node_risk'])
        self.assertEqual(r['correct_information_utility'],0)
    def test_wrong_leaf_and_valid_parent_have_distinct_risks(self):
        r=self.t.metrics(['a','c'],['g','a'])
        self.assertEqual(r['node_risk'],.5);self.assertEqual(r['leaf_risk'],1)
        self.assertEqual(r['leaf_coverage'],.5)
    def test_climbing_and_lca(self):
        p={'a':.45,'b':.4,'c':.15}
        self.assertEqual(self.t.climbing(p,.8),'g')
        self.assertEqual(self.t.climbing(p,.9),'root')
        self.assertEqual(self.t.flat(p,.8),'root')
        self.assertEqual(self.t.lca(['a','b']),'g')
        self.assertEqual(self.t.lca([]),'root')
    def test_bad_tree_and_probabilities(self):
        with self.assertRaises(ValueError):Taxonomy({'a':'b','b':'a','c':'root'})
        with self.assertRaises(ValueError):self.t.climbing({'a':1,'b':1,'c':1},.5)
    def test_two_stage_can_choose_another_branch_but_cannot_admit_a_low_scoring_singleton(self):
        p={'a':.31,'b':.29,'c':.4}
        self.assertEqual(self.t.climbing(p,.55),'root')
        self.assertEqual(self.t.two_stage(p,.55,.55),'g')
        self.assertEqual(self.t.two_stage(p,.4,.55),'c')
        self.assertEqual(self.t.two_stage({'a':.22,'b':.23,'c':.55},.7,.5),'root')
    def test_sample_size_and_exact_bounds(self):
        delta=.05/32
        self.assertGreater(upper_bound(0,125,delta),.05)
        self.assertLess(upper_bound(0,126,delta),.05)
        self.assertIsNone(upper_bound(0,0,delta))
        self.assertAlmostEqual(upper_bound(1,2,.05),math.sqrt(.95),places=10)
    def test_conformal_small_sample_and_empty_set(self):
        self.assertTrue(math.isinf(conformal_quantile([.1],.05)))
        self.assertEqual(lac_set({'a':.6,'b':.4},.5),['a'])
    def test_no_acceptance_not_certifiable(self):
        r=select_policy(self.t,['a'],{'reject':['root']})
        self.assertIsNone(r['selected'])
    def test_group_isolation_and_determinism(self):
        rows=[{'lesion_id':str(i//2),'dx':'a' if i<30 else 'b'} for i in range(60)]
        a=grouped_splits(rows,17);self.assertEqual(a,grouped_splits(rows,17))
        self.assertEqual(len(a),30)
        self.assertNotEqual(a,grouped_splits(rows,29))
        rows.append({'lesion_id':'0','dx':'b'})
        with self.assertRaises(ValueError):grouped_splits(rows,17)


if __name__=='__main__':unittest.main(verbosity=2)
