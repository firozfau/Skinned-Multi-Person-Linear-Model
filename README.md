# smpl
A Skinned Multi-Person Linear Model (SMPL) , SMPL is a realistic 3D model of the human body that is based on skinning and blend shapes and is learned from thousands of 3D body scans.

Unlike previous models, the pose-dependent blend shapes are a linear function of the elements of the pose rotation matrices. 
Traditional body modeling methods create unrealistic joint distortions, need extensive manual work, and lack compatibility with standard graphics software.
To solve this, we present the Skinned Multi-Person Linear (SMPL) model, which realistically captures diverse body shapes and natural pose deformations, including soft-tissue dynamics, and is easy to animate with existing tools.


##Generate:
python -m venv myenv
source myenv/bin/activate

pip install -r requirements.txt

pip install omegaconf

pip install loguru

## run code
python smplifyx/main.py --config cfg_files/fit_smplx.yaml \
    --data_folder DATA_FOLDER \
    --output_folder OUTPUT_FOLDER \
    --visualize=True \
    --model_folder MODEL_FOLDER \
    --vposer_ckpt VPOSER_FOLDER \
    --part_segm_fn smplx_parts_segm.pkl



## When just you want to run model which you already generate: 


## for single simulation run
python codeHub/frz.py --model-folder /Users/frzf7/Documents/www/python_server/computer_vision/smplx/smplx/models --model-type smplx --gender neutral --plot-joints True


## fo combind simulation run:
 python codeHub/main.py --model-folder /Users/frzf7/Documents/www/python_server/computer_vision/smplx/smplx/models --plot-joints true --genders neutral male female


## Important-Note:
'Before run this project you have to unzip models zip file (it is required )'
Smplx/ models.zp
