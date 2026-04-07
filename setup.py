from setuptools import setup, find_packages
setup(
    name='pymlutils',
    version='0.4.2',
    author='Shebin S Illikkal',
    author_email='Shebinsillikkal@gmail.com',
    description='Reusable ML utilities — preprocessing, evaluation, explainability',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/shebinsillikkal/pymlutils',
    packages=find_packages(),
    install_requires=['numpy>=1.24', 'pandas>=2.0', 'scikit-learn>=1.3',
                      'matplotlib>=3.7', 'shap>=0.43'],
    python_requires='>=3.9',
)
