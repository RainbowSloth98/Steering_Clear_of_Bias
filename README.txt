analyse_results.py
	Visualise and look at the features of SAEs


Asylex_data_clensing.py
	Took as input the original Asylex data, which we thoroughly cleaned.
	Produced our_train and our_test


full_acts_extract.py
			Was originally also used to produce the tpick tokds.
	Runs a model (BERT or bge) via nnsight, in order to extract activations for classifying
		Both just CLS token, but also pooling.
	Takes as input a tokenised dataset
	Produces frags(fragmented cls outputs)


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


mact3.py
	Replaced mact_trainer.py, mostly done by Gemini
	Does a stratified, reservoir sampling approach of the original
	


mact_analysis.py
			Now in old pys, replaced by seek_feat.py
	Runs the output from mact_trainer, letting us see and analyise the meaning of different activations

seek_feat.py
	Replaces mact_analysis, combined with show_tens
	Acts as the new dashboard


auto_feat_find_tester.py
	File that, in parallel, searches for a list of gendered words through all of the feature interps.
		This has been very productive in finding important features.


SAE_trainer.py
	Main file for training SAEs; Originally did them on layers 3,6 and 9, but we mostly stuck to just 6
		3,6,9 were the indicies, I was silly
	We started with standard ReLU, but switched to BatchTopK SAEs. Still need to fully make the switch in training.


train_classhead.py
	Takes final-layer activations as input, and trains a classifier layer for Asylex
		classifier can be either a single linear layer, or an MLP layer
		Though we mostly just stick to the linear one (for now).
	The output is a classifier that we use in steering_tester.py to see whether our steering effects the output.

steering_tester.py
					Moved to old, replaced with steering_refactored.py
		Now uses a modified ActDataset.
	The main place all the steering is done.
	Takes a TensorDataset with labels, as input and puts it through nnsight
	Also the main place where we use the classifier head that we trained.


nullspace_projector.py
	Using and implementing based on the Null it out paper
		/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Writeup/relevant_reading/null-it-out-aka-smtsmtprobe.pdf


pytester.py
	A file used specifically to test and check other bits of code.
		Currently testing if the INLP is working (spoiler, it isn't)

gender_classifier.py
	Used to test a reliable gender classifier for our task
		Should use more or less the same code from train_classhead.py






