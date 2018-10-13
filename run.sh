CUDA_VISIBLE_DEVICES=1 python train_model.py --name=separateW1 --mse-weight=1 --epochs=15&
CUDA_VISIBLE_DEVICES=2 python train_model.py --name=separateW01 --mse-weight=0.1 --epochs=15&
CUDA_VISIBLE_DEVICES=3 python train_model.py --name=separateW001 --mse-weight=0.01 --epochs=15&
CUDA_VISIBLE_DEVICES=1 python train_model.py --name=separateW0001 --mse-weight=0.001 --epochs=15&
CUDA_VISIBLE_DEVICES=2 python train_model.py --name=separateW00001 --mse-weight=0.0001 --epochs=15&
CUDA_VISIBLE_DEVICES=3 python train_model.py --name=separateW000001 --mse-weight=0.00001 --epochs=15&
wait
CUDA_VISIBLE_DEVICES=1 python benchmark_model.py --name=separateW1&
CUDA_VISIBLE_DEVICES=2 python benchmark_model.py --name=separateW01&
CUDA_VISIBLE_DEVICES=3 python benchmark_model.py --name=separateW001&
CUDA_VISIBLE_DEVICES=1 python benchmark_model.py --name=separateW0001&
CUDA_VISIBLE_DEVICES=2 python benchmark_model.py --name=separateW00001&
CUDA_VISIBLE_DEVICES=3 python benchmark_model.py --name=separateW000001&
