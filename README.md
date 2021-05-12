# PLOME:Pre-training with Misspelled Knowledge for Chinese Spelling Correction (ACL2021)
This repository provides the code and data of the work in ACL2021: *PLOME: Pre-training with Misspelled Knowledge for Chinese Spelling Correction*

**Requirements:**

- python3

- tensorflow1.14

- horovod

**Instructions:**

- Finetune: 

   train and evaluation file format: original sentence \t golden sentence 
   ```bash
   step1: cd finetune_src ; 
   step2: download the pretrained PLOME model and corpus from https://drive.google.com/file/d/1aip_siFdXynxMz6-2iopWvJqr5jtUu3F/view?usp=sharing ;
   step3: sh start.sh
   ```
   
 - Pre-train
   ```bash
   step1: cd pre_train_src ;
   step2: sh gen_train_tfrecords.sh ;
   step3: sh start.sh
   ```
   Our pre-trained model: https://drive.google.com/file/d/1aip_siFdXynxMz6-2iopWvJqr5jtUu3F/view?usp=sharing
