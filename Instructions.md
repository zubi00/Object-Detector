Implement of CenterNet on [visdrone2021](http://aiskyeye.com) dataset. The neck is modified to fpn with deconv.    
## Dependencies
- Python >= 3.6
- PyTorch >= 1.6
- opencv-python
- pycocotools
- numba


## Data

The data structure would look like:
```
data/
    visdrone/
        annotations/
        train/
        test/
        val/
```

Coco format visdrone2021 download from [google drive](https://drive.google.com/drive/folders/1FaXxOn349-YUsKa95G22etVlOf_Gj6rg?usp=sharing). 
You can also download the original dataset from http://aiskyeye.com and use the tools in src/tools to convert the format by yourself. 



## Train
python main.py --arch resnet18 --min_overlap 0.3 --gpus 0,1 --num_epochs 100 --lr_step 60,80 --batch_size 4 --lr 0.15625e-4 --exp_id <save_dir>


You can specify more parameters in src/opt.py.  
Results(weights and logs) will default save to exp/default if you dont specify --exp_id.  
Arch supports resnet18,resnet34,resnet50,resnet101,resnet152,res2net50,res2net101.  
If you scale batch_size, lr should scale too. 

## Test
python test.py --arch resnet18 --gpus 0 --load_model <path/to/weight_name.pt> --flip_test 



## Demo 
python demo.py --arch resnet18 --gpus 0 --load_model <path/to/weight_name.pt> --image <path/to/your_picture.jpg>


