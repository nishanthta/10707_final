Welcome to our repo!!

Herein is documented our work towards building the VitSigNet model, which was the result of our curiosity as to whether pose vectors could serves as a useful feature by themselves in the task of forgery detection in handwritten signatures. 

Clone this repo, and make sure to run `pip install -r requirements.txt`.
(plese note that you'll need an environemnt that has `python >=3.7,<3.11` for the `requirements.txt` to successfully install. For this, you can use `pyenv` or an equivalent tool.

Then, follow the instructions in `demo.ipynb` to see a test run of the models we have trained.

If you wish to train your own VitSigNet, use the `train_siam.py` script and make sure to make adjustments to the dataloader and/or filepaths as needed.


Credits to the authors of this [repo](https://github.com/jaehyunnn/ViTPose_pytorch.git) for their clean and efficient implementation of the the VitPose model, without which our work here would've been miles harder. If you have any suggestions on how we can improve our work, please feel to reach out on juvanjos [at] andrew.cmu.edu or nthumbav [at] andrew.cmu.edu !!
