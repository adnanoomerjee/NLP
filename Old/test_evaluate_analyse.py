import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def test(model):
	def device():
			if torch.backends.mps.is_available():
				device = torch.device("mps") 
			elif torch.cuda.is_available():
				device = torch.device('cuda')
			else:
				device = torch.device('cpu')
			return device
		
		
	device = device()

	def test(model, test_data, ground_truths):
		
		model.eval()

		y_preds = []
		y_true = []
		total_loss = 0.0

		with torch.no_grad():
			for i in range(len(test_data)):
				# Convert input to tensor and move to device
				inputs = test_data[i].to(device)

				# Get ground_truths
				y_true.append(ground_truths[i])

				# Forward pass
				outputs = model(inputs)

				# Predicted labels
				_, predicted = torch.max(outputs, 1)
				y_preds.append(predicted.item())

		return y_true, y_preds

	def evaluate(y_true, y_preds):
		# Calculate evaluation metrics
		precision, recall, f1, _ = precision_recall_fscore_support(
			y_true, y_preds, average=None)

		weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
			y_true, y_preds, average='weighted')

		accuracy = accuracy_score(y_true, y_preds)

		loss = nn.CrossEntropyLoss()(y_preds, y_true)

		metrics = {
			'precision': precision,
			'recall': recall,
			'f1': f1,
			'precision_weighted': weighted_precision,
			'recall_weighted': weighted_recall,
			'f1_weighted': weighted_f1,
			'accuracy': accuracy,
			'loss': loss
		}

		# Print evaluation metrics
		print('Precision per class:', precision)
		print('Recall per class:', recall)
		print('F1 score per class:', f1)
		print('Weighted precision:', weighted_precision)
		print('Weighted recall:', weighted_recall)
		print('Weighted F1 score:', weighted_f1)
		print('Accuracy:', accuracy)
		print('Loss:', loss)

		return metrics

	def analyse_errors(model, test_data, ground_truths):
		
		model.eval()

		with torch.no_grad():
			errors = []
			tp = torch.zeros(2).to(device) # true positives
			fp = torch.zeros(2).to(device) # false positives
			fn = torch.zeros(2).to(device) # false negatives
			for i in range(len(test_data)):
				# Convert input to tensor and move to device
				inputs = test_data[i].to(device)

				# Forward pass
				outputs = model(inputs)

				# Predicted labels
				_, predicted = torch.max(outputs, 1)
				if predicted.item() != ground_truths[i]:
					# Collect information about the error
					error_info = {}
					error_info['id'] = i
					error_info['Target sentence'] = test_data[i][1]
					error_info['Context sentence 1'] = test_data[i][0]
					error_info['Context sentence 2'] = test_data[i][2] 
					error_info['Predicted label'] = predicted.item()
					error_info['True label'] = ground_truths[i]

					errors.append(error_info)
		return errors

# Create instance of the model and load saved weights and move model to GPU, if available
model = 
model.to(device)

# Prepare test data and ground_truths
test_data = 
ground_truths = 

test_data_tensor = torch.tensor(test_data)
ground_truth_tensor = torch.tensor(ground_truths)

# Test Model
y_true, y_preds = test(model, test_data_tensor, ground_truth_tensor)

# Evaluate model
metrics = evaluate(y_true, y_preds)

# Analyse errors
erros = analyse_errors(model, test_data_tensor, ground_truth_tensor)


