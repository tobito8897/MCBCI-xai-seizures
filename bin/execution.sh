#python generate_windows_chb.py --class=train --overlap=0.5 --proportion=1
#python generate_windows_chb.py --class=test --overlap=0 --proportion=1
#python train_model_crossseizure.py --model=wang_1d --db=chb-mit --overlap=0.5
#python generate_windows_chb.py --class=test --overlap=0.5 --proportion=1
#python reevaluate_model_crossseizure.py --model=wang_1d --db=chb-mit --overlap=0.5

#python generate_windows_siena.py --class=train --overlap=0.5 --proportion=1
#python generate_windows_siena.py --class=test --overlap=0 --proportion=1
#python train_model_crossseizure.py --model=wang_1d --db=siena --overlap=0.5
#python generate_windows_siena.py --class=test --overlap=0.5 --proportion=1
#python reevaluate_model_crossseizure.py --model=wang_1d --db=siena --overlap=0.5

#python generate_windows_tusz.py --class=train --overlap=0.5 --proportion=1
#python generate_windows_tusz.py --class=test --overlap=0 --proportion=1
#python train_model_crossseizure.py --model=wang_1d --db=tusz --overlap=0.5
#python generate_windows_tusz.py --class=test --overlap=0.5 --proportion=1
#python reevaluate_model_crossseizure.py --model=wang_1d --db=tusz --overlap=0.5


#python generate_windows_chb.py --class=test --overlap=0 --proportion=1
#python generate_windows_siena.py --class=test --overlap=0 --proportion=1
#python generate_windows_tusz.py --class=test --overlap=0 --proportion=1
#python generate_windows_tusz.py --class=train --overlap=0.8 --proportion=1
#python generate_windows_tusz.py --class=train --overlap=0.8 --proportion=1
#python generate_windows_tusz.py --class=train --overlap=0.8 --proportion=1

#python generate_explanations_lime.py --db=chb-mit --model=wang_1d --overlap=0.8
#python generate_explanations_shap.py --db=chb-mit --model=wang_1d --overlap=0.8

#python generate_explanations_lime.py --db=siena --model=wang_1d --overlap=0.8
#python generate_explanations_shap.py --db=siena --model=wang_1d --overlap=0.8

#python generate_explanations_lime.py --db=tusz --model=wang_1d --overlap=0.8
#python generate_explanations_shap.py --db=tusz --model=wang_1d --overlap=0.8