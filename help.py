# -*- coding: utf-8 -*-
"""Help strings for key ML metrics."""

recall = """
The ratio of positive instances that are correctly detected by
the classifer (the True Positive Rate), i.e. the number of true positives
over the number of actual positives. High recall indicates finding all
positive instances, potentially at the expense of increasing the number of
false positves.
"""

precision = """
The number of true positive instances over the number of predicted
positives, i.e. how accurate the classifier is on a per-class basis.
High precision indicates positive predictions with high certainty,
potentially at the expense of missing some instances.
"""
