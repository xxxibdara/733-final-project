from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# load saved model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./tag_prediction/tag_model")
model = AutoModelForSeq2SeqLM.from_pretrained("./tag_prediction/tag_model")


# tokenize and preprocess input
def preprocess_input(text):
    inputs = tokenizer([text], max_length=512,
                       truncation=True, return_tensors="pt")
    return inputs

# generate output and decode tags list
def decode_output(inputs):
    output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10,
                            max_length=64)
    decoded_output = tokenizer.batch_decode(
        output, skip_special_tokens=True)[0]
    tags = list(set(decoded_output.strip().split(", ")))
    return tags


# main function to get tags
def get_tags(text):
    inputs = preprocess_input(text)
    tags = decode_output(inputs)
    return tags
