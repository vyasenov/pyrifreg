# pyrifreg

A Python package for Recentered Influence Function (RIF) regression analysis. This package provides tools for analyzing distributional effects in econometrics and data science applications.

## Installation

You can install the package using pip:

```bash
pip install pyrifreg
```

## Features

- Implementation of Recentered Influence Function (RIF) regression
- Support for various distributional statistics (mean, quantiles, variance, etc.)
- Easy-to-use API for regression analysis
- Integration with pandas and scikit-learn

## Quick Start

```python
import numpy as np
import pandas as pd
from pyrifreg import RIFRegression

# Create sample data
X = np.random.randn(1000, 2)
y = np.random.randn(1000)

# Initialize and fit RIF regression
rif_reg = RIFRegression(statistic='mean')
rif_reg.fit(X, y)

# Get regression results
results = rif_reg.summary()

# For Gini coefficient
gini_rif = RIFRegression(statistic='gini')

# For IQR
iqr_rif = RIFRegression(statistic='iqr')

# For entropy
entropy_rif = RIFRegression(statistic='entropy')

# For quantiles (as before)
median_rif = RIFRegression(statistic='quantile', q=0.5)
```

## Documentation

For detailed documentation and examples, please visit our [documentation page](https://github.com/yourusername/pyrifreg).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
