The `score_site` library is a tool for performing Multi-Criteria
Decision Analysis (MCDA) of wave and tidal market opportunities in the
U.S.. This package was developed under the U.S. Department of Energy's
'Early Market Opportunity Hot Spot Identification' project. The
results of this work are available in the MHK Energy Early Market Site
Identification and Ranking Methodology reports.

The package was used to analyze and create the tables presented in
those reports. At the most basic level, it is meant to perform MCDA on
two types of spatial data:

1) Site data, which is a single point with scorable 'attributes'
(e.g. Yakutat, Alaska has a 'load' and a 'cost of energy'.)

2) Spatially distributed information (spatial attributes) associated
with that site (e.g. wave energy density and water depth).

This package provides a simple 'scoring model' approach to MCDA
analysis. First, a user imports the package and loads data from a
specially-formatted data file:

    # Load the package,
    import score_site as ss

    # Load the data,
    dat = ss.load_excel('A_file_containing_Site_and_Resource_data.xlsx')

The user can then construct a scoring model that scores the columns of
the data:

    score_model = ss.SumModel(load=ss.scorers.Linear([100,10000]),
                              resource=ss.scorers.Linear([10, 30]),
                              )

Finally, the results are calculated by applying the scoring model to
the data:

    results = score_model(dat)

    # Save the results,
    results.to_excel('Some_results.xlsx')


Installation
-------

This package is not meant for *install*, per se. Rather, it should be
used on the specific data files for which it was created. As such, it
is recommended that you simply place this package in a working
directory containing the scripts that will utilize it. To do this
either unzip the repositories'
[zip file](http://github.com/lkilcher/score_site/archive/master.zip)
into a `score_site` folder in the working directory, or `cd` into that working directory, and
type:

    git clone http://github.com/lkilcher/score_site.git

The dependencies of this package can be found in the
`requirements.txt` file. To install the *dependencies* for this
package, simply `cd` into the score_site folder, and do:

    pip install -r requirements.txt
