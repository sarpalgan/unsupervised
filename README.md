# How to use

*1. Create and activate a virtual environment called `termin`*
```bash
conda create --name termin python=3.9
conda activate termin
```

*2. Install the requirements*
```bash
pip install -r requirements.txt
```

*3. Open python from the anaconda prompt by running the following:*
```bash
python
```

*4. Import the library:*
```bash
from model import Do
```

*5. Run one or many of the following lines depending on your request:*
```bash
Do.Preprocessing()
Do().Cleaning()
Do().Prediction(mode='test')
```

