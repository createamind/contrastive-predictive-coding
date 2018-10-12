CUDA_VISIBLE_DEVICES=1 python train_model.py --name=cpc1 --mse-weight=1 --epochs=15&
CUDA_VISIBLE_DEVICES=2 python train_model.py --name=cpc01 --mse-weight=0.1 --epochs=15&
CUDA_VISIBLE_DEVICES=3 python train_model.py --name=cpc001 --mse-weight=0.01 --epochs=15&
CUDA_VISIBLE_DEVICES=1 python train_model.py --name=cpc0001 --mse-weight=0.001 --epochs=15&
CUDA_VISIBLE_DEVICES=2 python train_model.py --name=cpc00001 --mse-weight=0.0001 --epochs=15&
CUDA_VISIBLE_DEVICES=3 python train_model.py --name=cpc000001 --mse-weight=0.00001 --epochs=15&
wait
CUDA_VISIBLE_DEVICES=1 python benchmark_model.py --name=cpc1&
CUDA_VISIBLE_DEVICES=2 python benchmark_model.py --name=cpc01&
CUDA_VISIBLE_DEVICES=3 python benchmark_model.py --name=cpc001&
CUDA_VISIBLE_DEVICES=1 python benchmark_model.py --name=cpc0001&
CUDA_VISIBLE_DEVICES=2 python benchmark_model.py --name=cpc00001&
CUDA_VISIBLE_DEVICES=3 python benchmark_model.py --name=cpc000001&
