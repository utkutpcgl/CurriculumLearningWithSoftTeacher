# Important commands
- Run and build docker with proper names, volumes and gpus.
docker build -t soft/teacher:v1 .
docker run -it --shm-size=14gb --gpus '"device=0,1"' -v /raid/utku/:/workspace -t deneme:latest does the job.
- Run soft teacher only on 2nd fold with 5 percent data only for 20000 iterations for easier testing.
bash tools/dist_train_partially.sh semi 2 5 4 --resume-from work_dirs/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k/5/2/iter_8000.pth