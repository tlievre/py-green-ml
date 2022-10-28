from setuptools import setup

setup(
        name="GreenML",
        version="0.0.1",
        author="Renan Manceaux, Quentin Faidide, Lorys Debbah, Thomas Lievre",
        packages=["models", "preprocessing", "runners"],
        url="https://gitlab.com/m2-ssd/green-ml/pygreen_ml.git",
        description="Computes Machine Learning algorithms' power consumption",
        license='LICENSE',
        install_requires=[
            'pandas',
            'seaborn',
            'scikit-learn',
            'tqdm',
            'torch'
            ],
        tests_require=['pytest'],
        package_data={
            '': ['models.json']
            }

)
