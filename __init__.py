"""
The score_site package is a tool for performing Multi-Criteria
Decision Analysis (MCDA) of data with two types of spatial data:

1) Site data, which is a single point with scorable 'attributes'
(e.g. Yakutat, Alaska has a 'load' and a 'cost of energy'.)

2) Spatially distributed information (spatial attributes) associated
with that site (e.g. wave energy density and water depth).

This interface package provides access to the most commonly used
components of the package.

Data Types
..........

* :class:`.HotSpotCollection` : The primary data type of the
  score_site package. This wraps the site and resource data into one
  data object that can be manipulated and analyzed in a consistent and
  'Pythonic' way.

* :class:`.HotSpot` : This is essentially the same as
  :class:`.HotSpotCollection`, but for a single point. It contains a
  few extra shortcuts and functions that are useful for single sites.

Scoring
.......

* :mod:`scorers` : The module that contains the scoring classes.
  Scorers score an individual attribute (column) of a
  :class:`.HotSpotCollection`.

Models are initialized with a set of 'scorers' for scoring each column
of the data and define a method for aggregating scores to compute a
'score_total'.

* :class:`.ProdModel` : Aggregates scores using the 'product method'.
  :class:`.HotSpotCollection` object.

* :class:`.SumModel` : a 'sum model' class for scoring a
  :class:`.HotSpotCollection` object.

* :class:`.MultiModel` : A scoring model class that is composed of
  several scoring models, the scores from the highest individual model
  are returned as the aggregate value.

Miscellaneous
.............

* :func:`.load_excel` : A function for loading data from *properly
formatted* Excel files into a :class:`.HotSpotCollection`. See this
functions documentation for information on this format.

* cmaps : A dictionary of useful colormaps.

Examples
--------

::

    # Load the package,
    import score_site as ss

    # Load the data,
    dat = ss.load_excel('A_file_containing_Site_and_Resource_data.xlsx')

    # Create a scoring model,
    score_model = ss.SumModel(load=ss.scorers.Linear([100,10000]),
                              resource=ss.scorers.Linear([10, 30]),
                              )

    # Score the data,
    results = score_model(dat)

    # Save the results,
    results.to_excel('Some_results.xlsx')

"""

from base import HotSpot, HotSpotCollection
import scorers
from models import ProdModel, SumModel, MultiModel
from io import load_excel
from base import cmaps
