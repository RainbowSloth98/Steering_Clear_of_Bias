analyse_results.py
	Visualise and look at the features of SAEs


frag_combiner.py
	Takes as input fragmented activations that need to be combined
		This is also accross all the different experiments
	Combine together into a single tensor and store as all_cls_acts


gemini_gender_studies.py
	Takes our_train, and runs Gemini API to generate gender data
	Output was a dirty dataset that contained gender info but needed further cleaning:
		Bad entries (Gemini did not process)
		Remove indicies past 23121 (did not process these by accident)
		Remove the "Unclear" entries.
		Reset the index for consistency
	The same was done for the labels as well


nullspace_projector.py
	Using and implementing based on the Null it out paper
		/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Writeup/relevant_reading/null-it-out-aka-smtsmtprobe.pdf


Asylex_data_clensing.py
	Took as input the original Asylex data, which we thoroughly cleaned.
	Produced our_train and our_test


full_acts_extract.py
			Was originally also used to produce the tpick tokds.
	Runs a model (BERT or bge) via nnsight, in order to extract activations for classifying
		Both just CLS token, but also pooling.
	Takes as input a tokenised dataset
	Produces frags(fragmented cls outputs)


gender_classifier.py
	Used to test a reliable gender classifier for our task
		Should use more or less the same code from task_classifier.py

task_classifier.py
	Takes final-layer activations as input, and trains a classifier layer for Asylex
		classifier can be either a single linear layer, or an MLP layer
		Though we mostly just stick to the linear one (for now).
	The output is a classifier that we use in steering_tester.py to see whether our steering effects the output.


SAE_trainer.py
	Main file for training SAEs; Originally did them on layers 3,6 and 9, but we mostly stuck to just 6
		3,6,9 were the indicies, I was silly
	We started with standard ReLU, but switched to BatchTopK SAEs. Still need to fully make the switch in training.


inlp_tester.py
	File that trains the INLP transform and tests to see how well it works
		Worked somewhat well at layer 6, but got the opposite effect by the end


auto_feat_find_tester.py
	File that, in parallel, searches for a list of gendered words through all of the feature interps.
		This has been very productive in finding important features.
					ONLY SELECTED FEATURES THAT ACTIVATED ON PREMADE LIST OF GENDERED WORDS


mact3.py
	Replaced mact_trainer.py, mostly done by Gemini
	Does a stratified, reservoir sampling approach of the original


steering_tester.py
					Moved to old, replaced with steering_refactored_2.py
		Now uses a modified ActDataset.
	The main place all the steering is done.
	Takes a TensorDataset with labels, as input and puts it through nnsight
	Also the main place where we use the classifier head that we trained.


strah_experiment.py
	Found main male and female feat (4364 and 1562)
	Devised a plan for finding feats and steering
		Use the current partial counterfact data to do A-B
			Should reveal the important features, for our dataset.
		Afterwards aggregate via one of two methods
			Aggregate all of them together into one transform
			Cluster them into subtypes, and do CosSim to decide which transform to use
	Should get transforms we can apply to the feature vectors that swap between M and F















