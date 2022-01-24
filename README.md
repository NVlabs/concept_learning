# concept_learning

**Downloading Object Assets and Pre-trained Models:**

Download [data.tar.gz](https://drive.google.com/file/d/1h-jcEI-SArBFR4FO4uL09jUYle6zRPjQ) and place it under the `data` folder. Unzip it with the following command:

```Shell
tar zxvf data.tar.gz
```

**Functionality:**

1. a) Generate passive data using Isaac Gym: 

    `python datasets/generate_concept_data.py --headless --cuda --envs <num_envs> --samples <num_samples>`

Options:

- `--headless`: starts data generation without simulator visualizer;
- `--cuda`: activates GPU computation;
- `<num_envs>`: number of Gym environments to generate data in parallel;
- `<num_samples>`: number of samples to collect per environment.

1. b) Generate active data:

    `python datasets/collect_human_data.py --config <config_path> --concept <concept> --simulated --active_samples <active_samples> --passive_samples <passive_samples> --batch_size <batch_size> --objective <objective> --warmstart <warmstart_samples> --mining <mining_samples>`

Options:

- `<concept>`: concept to ask for queries about ("above180", "above45", "abovebb", "near", "upright", "upright45", "alignedhorizontal", "alignedvertical", "front", "front45", "ontop");
- `<config_path>`: path to the `/configs` folder configuration file that specifies training parameters;
- `--simulated`: uses the oracle human to label the queries, otherwise uses real human input via interface;
- `<active_samples>`: number of samples to collect from the human actively (label queries);
- `<passive_samples>`: number of samples to collect from the human passively (demo queries);
- `<batch_size>`: how often to retrain the active learning model;
- `<objective>`: active learning objective ("random", "min", "max", "confusion", "allrand", "confrand")
- `--warmstart`: number of demonstration queries to use for warmstarting the model;
- `--mining`: number of queries that use mining.

2. Label data using ground truth or learned concept:

    `python datasets/label_concept_data.py --concept <concept> --concept_model <concept_model>`

Options:

- `<concept>`: name of desired concept ("above180", "above45", "abovebb", "near", "upright", "upright45", "alignedhorizontal", "alignedvertical", "front", "front45", "ontop");
- `<concept_model>`: name of ".pt" model weights file.

3. Create test-train split files for oracle demonstration data:

    `python datasets/split_data.py --concept <concept>`

Options:

- `<concept>`: name of desired concept ("above180", "above45", "abovebb", "near", "upright", "upright45", "alignedhorizontal", "alignedvertical", "front", "front45", "ontop");

4. Train concept:

    `python train/train_oracle_concept.py --concept_dir <concept_dir> --config <config_path> --train_amt <train_amount>`

Options:

- `<concept_dir>`: concept's label directory;
- `<config_path>`: path to the `/configs` folder configuration file that specifies training parameters;
- `<train_amt>`: data amount used for training (optional: if left out, uses all data);

5. Evaluate concept:

    `python train/evaluate_concept.py --concept_dir <concept_dir> --config <config_path> --concept_model <concept_model>`

Options:

- `<concept_dir>`: concept's label directory;
- `<config_path>`: path to the `/configs` folder configuration file that specifies evaluation parameters;
- `<concept_model>`: model to use for evaluation;

Note: The above functionalities are replicated for ngc job scheduling in the `/scripts` folder.

**Passive Example (Demo Queries):**

1. Generate Isaac Gym data: `python datasets/generate_concept_data.py --headless --cuda --envs 100 --samples 10000`
2. Label data with ground truth: `python datasets/label_concept_data.py --concept "above45"`
3. Split data in a balanced way (demo queries): `python datasets/split_data.py --concept "above45"`
4. Train low-dimensional concept: `python train/train_oracle_concept.py --concept_dir "above45" --config '/../../configs/rawstate_oracle.yaml' --train_amt 500`
5. Train baseline: `python train/train_oracle_concept.py --concept_dir "above45" --config '/../../configs/pointcloud_oracle.yaml' --train_amt 500`
6. Train handcrafted: `python train/train_oracle_concept.py --concept_dir "above45" --config '/../../configs/pointcloud_oracle.yaml'`
7. Label using the low-dimensional concept: `python datasets/label_concept_data.py --concept "above45" --concept_model 'oracle_rawstate_500_0.pt'`
8. Train ours: `python train/train_oracle_concept.py --concept_dir above45_500 --config '/../../configs/pointcloud_oracle.yaml'`
9. Evaluate model: `python train/evaluate_concept.py --concept_dir "above45" --config '/../../configs/pointcloud_oracle.yaml' --concept_model 'oracle_pointcloud_g500_0.pt'`

**Active Example (Label Queries):**

1. Generate active learning data: `python datasets/collect_human_data.py --config '/../../configs/rawstate_AL.yaml' --concept "above45" --simulated --active_samples 1000 --passive_samples 0 --batch_size 100 --objective 'confrand' --warmstart 0 --mining 1000`
2. Train low-dimensional concept: `python train/train_human_concept.py --concept "above45" --config '/../../configs/rawstate_human.yaml' --train_amt 1000 --strategy 'confrandmine'`
3. Train baseline: `python train/train_human_concept.py --concept "anove45" --config '/../../configs/pointcloud_human.yaml' --train_amt 1000 --strategy 'randomgt'`
4. Label using the low-dimensional concept: `python datasets/label_concept_data.py --concept "above45" --concept_model 'confrandgt_rawstate_1000_0.pt'`
5. Train ours: `python train/train_oracle_concept.py --concept_dir above45_confrandgtmine500 --config '/../../configs/pointcloud_oracle.yaml'`
6. Evaluate model: `python train/evaluate_concept.py --concept_dir "above45" --config '/../../configs/pointcloud_oracle.yaml' --concept_model 'confrandgtmine_pointcloud_g500_0.pt'`
