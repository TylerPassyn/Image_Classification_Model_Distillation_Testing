Epoch 1/5:   0%|          | 0/1000 [00:00<?, ?batch/s]Epoch 1/5:   0%|          | 0/1000 [12:55<?, ?batch/s]
Epoch 2/5:   0%|          | 0/1000 [00:00<?, ?batch/s]Epoch 2/5:   0%|          | 0/1000 [12:58<?, ?batch/s]
Epoch 3/5:   0%|          | 0/1000 [00:00<?, ?batch/s]Epoch 3/5:   0%|          | 0/1000 [12:58<?, ?batch/s]
Epoch 4/5:   0%|          | 0/1000 [00:00<?, ?batch/s]Epoch 4/5:   0%|          | 0/1000 [12:58<?, ?batch/s]
Epoch 5/5:   0%|          | 0/1000 [00:00<?, ?batch/s]Epoch 5/5:   0%|          | 0/1000 [13:03<?, ?batch/s]
Traceback (most recent call last):
  File "/deac/csc/classes/csc373/passta23/model_distillation/Distillation_Tests.py", line 76, in <module>
    main()
  File "/deac/csc/classes/csc373/passta23/model_distillation/Distillation_Tests.py", line 63, in main
    distilled_model = distiller.distill(logits_train, extracted_features, num_epochs=5, learning_rate=0.0001)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/passta23/model_distillation/Image_Distiller.py", line 81, in distill
    
AttributeError: 'Distill_Model_VIT_to_VIT' object has no attribute 'extracted_features'
