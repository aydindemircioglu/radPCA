
# radPCA

This repository contains the code for the paper
'Benchmarking Feature Projection Methods in Radiomics'
(to appear).


## Experiment

The results are in ./results, the figures for the paper are in ./paper.

To re-generate the results, first create a virtual environment, and
install the requirements by

```pip3 install -r requirements.txt```

Modify ./experiment.py to your needs, the main issue might be the number of
parallel jobs, which is current set to 30. Change 30 in the line

```    results = Parallel(n_jobs=30)(```

near the end of the file. Then start it via

```python3 ./experiment.py```

It took over a week with my setup (thanks, Boruta!).

*IMPORTANT*: The experiment will cache the feature reduction methods,
depending on the settings either to memory or to disk.
You can control that with the `cache_to = "memory"` flag in the script.
ENSURE THAT YOU HAVE AT LEAST 1 TB FREE HARD DISK SPACE. I cannot
remember how much it took, but I think I saw at least 500 GB. But maybe that
was when I tried SuperPCA (and failed). If caching to memory, then
better have 256GB RAM, though I think even 64GB could work. The
failed SuperPCA experiment might have distorted my view.


Unfortunately, the timings during CV were not saved properly, so we recompute them.
Execute ./getTimings.py for this.

Then, continue to evaluate the results:

```python3 ./evaluate.py```

Note that you can evaluate without generating, since the results are
store within this repository.


## Software

Not all software is available as a pip module, therefore I used:

### PyMRMRe

- https://github.com/bhklab/PymRMRe

However, pymrmre is broken and is currently not updated.
Therefore install PymRMRe from the provided folder,
by calling python3 ./setup.py install.

### SRP/SuperPCA

- https://github.com/bghojogh/Principal-Component-Analysis/blob/master/PCA_SPCA/my_dual_supervised_PCA.py

This is not installable as a pip, so I provided
the right version here. No install is necessary, since this is a single
file ./SRP.py that gets loaded directly.



# License

Datasets are licensed by their respective owners.


## MIT License

Copyright (c) 2023 aydin demircioglu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



#
