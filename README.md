# Running LightGBM in parallel on Azure Machine Learning

Running LightGBM in parallel requires invoking the `lightgbm` command line tool. Running this on Azure Machine Learning (AML) in an AML Compute cluster can make it easier for you by managing the MPI cluster for you.

In this example, we first set up an [AML Environment](notebooks/00-Environment_Setup.ipynb) and then [submit a job to the cluster](/notebooks/01-LightGBM_Submission.ipynb)

