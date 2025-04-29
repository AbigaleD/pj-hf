# transformerTestClasses.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)

# transformerTestClasses.py
# ----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import testClasses
import util

from tokenizers import Tokenizer
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.decoders

from transformers import pipeline
import datasets
import torch


# Simple test case which evals an arbitrary piece of python code.
# The test is correct if the output of the code given the student's
# solution matches that of the instructor's.
class EvalTest(testClasses.TestCase):

    def __init__(self, name, question, testDict):
        super(EvalTest, self).__init__(name, question, testDict)
        self.preamble = compile(testDict.get('preamble', ""), "%s.preamble" % self.getPath(), 'exec')
        self.test = compile(testDict['test'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        # exec self.preamble in bindings
        exec(self.preamble, bindings)
        return str(eval(self.test, bindings))

    def execute(self, grades, moduleDict, solutionDict):
        result = self.evalCode(moduleDict)
        if result == solutionDict['result']:
            grades.addMessage('PASS: %s' % self.path)
            grades.addMessage('\t%s' % self.success)
            return True
        else:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\tstudent result: "%s"' % result)
            grades.addMessage('\tcorrect result: "%s"' % solutionDict['result'])

        return False

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')

            handle.write('result: "%s"\n' % self.evalCode(moduleDict))
        return True


# Hidden test case checks the md5 of the result. Student can view
# the test case but not the plain text of the solution.
class HiddenTest(EvalTest):

    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        # exec self.preamble in bindings
        exec(self.preamble, bindings)
        return util.md5(eval(self.test, bindings))

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# The hash of the result of evaluating the test must equal the below.\n')

            handle.write('result: "%s"\n' % self.evalCode(moduleDict))
        return True


# Test case that requires student to raise an exception.
class ExceptionTest(EvalTest):

    def execute(self, grades, moduleDict, solutionDict):
        try:
            result = self.evalCode(moduleDict)
        except Exception as inst:
            if str(type(inst)) == solutionDict['result']:
                grades.addMessage('PASS: %s' % self.path)
                grades.addMessage('\t%s' % self.success)
                return True
            raise inst
        
        grades.addMessage('FAIL: %s' % self.path)
        grades.addMessage('\t%s' % self.failure)
        grades.addMessage('\tstudent result: "%s"' % result)
        grades.addMessage('\tcorrect result: "%s"' % solutionDict['result'])

        return False

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# The result of evaluating the test must raise the following exception.\n')

            try:
                result = self.evalCode(moduleDict)
            except Exception as inst:
                result = str(type(inst))
            else:
                raise RuntimeError('Use ExceptionTest but no exception raised.')

            handle.write('result: "%s"\n' % result)
        return True


class VocabTest(testClasses.TestCase):

    def __init__(self, name, question, testDict):
        super(VocabTest, self).__init__(name, question, testDict)
        self.vocab = compile(testDict['vocab'], "%s.test" % self.getPath(), 'eval')
        self.merges = compile(testDict['merges'], "%s.test" % self.getPath(), 'eval')
        self.sentences = compile(testDict['sentences'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

    def getEncoder(self, moduleDict):
        bindings = dict(moduleDict)
        clean_vocab = bindings["tokenizer"].clean_vocab

        vocab = eval(self.vocab, bindings)
        merges = eval(self.merges, bindings)
        clean_vocab(vocab, merges)

        tokenizer = Tokenizer(tokenizers.models.BPE(vocab, merges))
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space = False)
        tokenizer.decoder = tokenizers.decoders.ByteLevel()

        return tokenizer.encode
    
    def sanity_check(self, grades, moduleDict):
        bindings = dict(moduleDict)
        clean_vocab = bindings["tokenizer"].clean_vocab

        old_vocab = eval(self.vocab, bindings)
        old_merges = eval(self.merges, bindings)

        vocab = eval(self.vocab, bindings)
        merges = eval(self.merges, bindings)
        clean_vocab(vocab, merges)

        # vocab items must be sequential
        vocab_items = sorted(vocab.items(), key=lambda x: x[1])
        j = -1
        for i, (k, v) in enumerate(vocab_items):
            if i != v:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\tVocab index must be sequential.')
                grades.addMessage('\tThe index of %s should be %d, but get %d instead.' % (k, i, v))
                return False
            if k not in old_vocab:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\tCleaned vocab must be a subset of the original vocab.')
                grades.addMessage('\tToken %s does not exist in the original vocab.' % (k))
                return False
            if old_vocab[k] <= j:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\tCleaned vocab must follow the order of the original vocab.')
                grades.addMessage('\tToken %s has index %d in the original vocab, which is no larger than the previous one %d.' % (k, old_vocab[k], j))
                return False
            j = old_vocab[k]

        # merges should not repeat
        counter = util.Counter()
        j = -1
        for merge in merges:
            counter[merge] += 1
            if counter[merge] > 1:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\tMerge %s appears multiple times.' % (str(merge)))
                return False
            if merge not in old_merges:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\tMerge %s does not appear in orginal merges.' % (str(merge)))
                return False
            idx = old_merges.index(merge)
            if idx <= j:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\tMerge %s has index %d in the original merges, which does not follow the original order.' % (str(merge), idx))
                return False
            j = idx
        
        return True

    def execute(self, grades, moduleDict, solutionDict):
        if not self.sanity_check(grades, moduleDict):
            return False

        encoder = self.getEncoder(moduleDict)
        sentences = eval(self.sentences)
        results = eval(solutionDict['results'])

        for sentence, result in zip(sentences, results):

            output = ' '.join(encoder(sentence).tokens)

            if output != result:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\t%s' % self.failure)
                grades.addMessage('\t      sentence: "%s"' % sentence)
                grades.addMessage('\tstudent result: "%s"' % output)
                grades.addMessage('\tcorrect result: "%s"' % result)
                return False
            
        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\t%s' % self.success)
        return True

    def writeSolution(self, moduleDict, filePath):
        encoder = self.getEncoder(moduleDict)
        sentences = eval(self.sentences)
        results = [' '.join(encoder(sentence).tokens) for sentence in sentences]

        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# The tokenization of each sentence must equal the below.\n')

            handle.write('results: """\n%s\n"""\n' % str(results))
        return True


class ClassificationTest(testClasses.TestCase):

    def __init__(self, name, question, testDict):
        super(ClassificationTest, self).__init__(name, question, testDict)
        self.ds_kwargs = compile(testDict['ds_kwargs'], "%s.test" % self.getPath(), 'eval')
        self.num_samples = int(testDict['num_samples'])
        self.thresholds = [float(t) for t in testDict['thresholds'].split()]
        self.success = testDict['success']
        self.failure = testDict['failure']

        self.random = util.random.Random()
        self.random.seed(testDict['random_seed'])

    def execute(self, grades, moduleDict, solutionDict):
        # load the module
        bindings = dict(moduleDict)
        get_topic_classification_pipeline = bindings["classification"].get_topic_classification_pipeline

        pipe = get_topic_classification_pipeline()

        # sanity check
        sample = "Would the US constitution be changed if the admendment received 2/3 of the popular vote?"
        output = pipe(sample)
        if not isinstance(output, dict):
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % sample)
            grades.addMessage('\tstudent result: "%s"' % output)
            grades.addMessage('\tThe output of the pipeline must be a dictionary.')
            return False

        if "label" not in output or "score" not in output:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % sample)
            grades.addMessage('\tstudent result: "%s"' % output)
            grades.addMessage('\tThe output of the pipeline must contain "label" and "score".')
            return False
        
        if not isinstance(output["label"], str):
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % sample)
            grades.addMessage('\tstudent result: "%s"' % output)
            grades.addMessage('\tThe label of the output must be a string.')
            return False
        
        if not isinstance(output["score"], float):
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % sample)
            grades.addMessage('\tstudent result: "%s"' % output)
            grades.addMessage('\tThe score of the output must be a float.')
            return False
        
        if output["score"] < 0 or output["score"] > 1:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % sample)
            grades.addMessage('\tstudent result: "%s"' % output)
            grades.addMessage('\tThe score of the output must be in [0, 1].')
            return False
        
        if output["label"] not in ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % sample)
            grades.addMessage('\tstudent result: "%s"' % output)
            grades.addMessage('\tThe label of the output must be one of the following: "Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government".')
            return False
        
        # test the pipeline
        ds_kwargs = eval(self.ds_kwargs, bindings)
        ds = datasets.load_dataset(**ds_kwargs)
        ds = ds.select(self.random.sample(range(len(ds)), self.num_samples))
        labels = ds.features['topic'].names

        cnts = [0, 0]
        failed_sample = None
        for data in ds:
            text = data['question_title']
            result = pipe(text)

            if labels[data['topic']] != result["label"]:
                cnts[1] += 1
                failed_sample = (text, result["label"], labels[data['topic']])
            
            cnts[0] += 1

        accuracy = 1 - cnts[1] / cnts[0]

        points = 0
        for threshold in self.thresholds:
            if accuracy >= threshold:
                points += 1
        grades.addPoints(points)
        if points >= len(self.thresholds):
            grades.addMessage('PASS: %s' % self.path)
            grades.addMessage('\t%s' % self.success)
        else:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % failed_sample[0])
            grades.addMessage('\tstudent result: "%s"' % failed_sample[1])
            grades.addMessage('\tcorrect result: "%s"' % failed_sample[2])
        grades.addMessage('\taccuracy: %s' % round(accuracy, 2))
        grades.addMessage('\tthresholds: %s' % str(tuple(round(threshold, 2) for threshold in self.thresholds)))

        return True

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# File intentionally blank.\n')
        return True


class EvaluationTest(testClasses.TestCase):

    def __init__(self, name, question, testDict):
        super(EvaluationTest, self).__init__(name, question, testDict)
        self.preds = compile(testDict['preds'], "%s.test" % self.getPath(), 'eval')
        self.golds = compile(testDict['golds'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

    def execute(self, grades, moduleDict, solutionDict):
        # load the module
        bindings = dict(moduleDict)
        get_macro_f1_metric = bindings["evaluation"].get_macro_f1_metric

        metric = get_macro_f1_metric()

        preds = eval(self.preds, bindings)
        golds = eval(self.golds, bindings)
        results = eval(solutionDict['results'])

        for pred, gold, result in zip(preds, golds, results):
            output = metric(pred, gold)

            if not isinstance(output, float) or abs(output - result) > 1e-3:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\t%s' % self.failure)
                grades.addMessage('\t      preds: "%s"' % pred)
                grades.addMessage('\t      golds: "%s"' % gold)
                grades.addMessage('\tstudent result: "%s"' % output)
                grades.addMessage('\tcorrect result: "%s"' % result)
                return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\t%s' % self.success)

        return True

    def writeSolution(self, moduleDict, filePath):
        bindings = dict(moduleDict)
        get_macro_f1_metric = bindings["evaluation"].get_macro_f1_metric

        metric = get_macro_f1_metric()

        preds = eval(self.preds, bindings)
        golds = eval(self.golds, bindings)
        results = [round(metric(pred, gold), 4) for pred, gold in zip(preds, golds)]

        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# The macro F1 score of each pair of preds and golds must equal the below.\n')

            handle.write('results: "%s"\n' % str(results))

        return True


class ChatTest(testClasses.TestCase):

    def __init__(self, name, question, testDict):
        super(ChatTest, self).__init__(name, question, testDict)
        self.dialogs = compile(testDict['dialogs'], "%s.test" % self.getPath(), 'eval')
        self.keywords = compile(testDict['keywords'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

    def is_valid_tokenizer(self, grades, dialog, pipe) -> bool:
        tokenized = pipe.tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)

        if not tokenized.startswith("<|im_start|>system"):
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      dialog: "%s"' % dialog)
            grades.addMessage('\tstudent result: "%s"' % tokenized)
            grades.addMessage('\tThe tokenized result must start with a system message.')
            return False
        
        if tokenized.count("<|im_start|>system") >= 2:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      dialog: "%s"' % dialog)
            grades.addMessage('\tstudent result: "%s"' % tokenized)
            grades.addMessage('\tThe tokenized result must only contain one system message.')
            return False
        
        for message in dialog:
            if message["content"] not in tokenized:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\t%s' % self.failure)
                grades.addMessage('\t      dialog: "%s"' % dialog)
                grades.addMessage('\tstudent result: "%s"' % tokenized)
                grades.addMessage('\tThe tokenized result must contain all messages.')
                return False

        return True

    def execute(self, grades, moduleDict, solutionDict):
        # load the module
        bindings = dict(moduleDict)
        get_chat_template = bindings["generation"].get_chat_template

        dialogs = eval(self.dialogs, bindings)
        keywords = eval(self.keywords, bindings)
        pipe = pipeline(model="HuggingFaceTB/SmolLM2-360M-Instruct")

        template = get_chat_template()

        # sanity check
        if not isinstance(template, str):
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      template: "%s"' % template)
            grades.addMessage('\tThe chat template must be a string.')
            return False

        pipe.tokenizer.chat_template = template

        for dialog, keyword in zip(dialogs, keywords):
            if not self.is_valid_tokenizer(grades, dialog, pipe):
                return False

            output = pipe(dialog, max_new_tokens=128)
            output = output[0]["generated_text"][-1]["content"]

            if keyword.lower() not in output.lower():
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\t%s' % self.failure)
                grades.addMessage('\t      question: "%s"' % dialog[-1]["content"])
                grades.addMessage('\tstudent result: "%s"' % output)
                return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\t%s' % self.success)

        return True
    
    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# File intentionally blank.\n')
        return True


class RefusalTest(testClasses.TestCase):

    def __init__(self, name, question, testDict):
        super(RefusalTest, self).__init__(name, question, testDict)
        self.dialogs = compile(testDict['dialogs'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

        detect_model = testDict.get('detect_model', None)
        self.pipe = pipeline(model=detect_model) if detect_model else None
    
    def is_refusal(self, response: str) -> bool:
        """
        Check if the response contains a refusal.
        """
        if self.pipe:
            return self.pipe(response)[0]['label'].lower() != 'normal'

        keywords = ["As an AI", "I don't", "I cannot", "I am not", "I won't", "I will not", "I cannot", "I do not", "I can't", "not to"]
        return any(keyword in response for keyword in keywords)

    def execute(self, grades, moduleDict, solutionDict):
        # load the module
        bindings = dict(moduleDict)
        get_chat_template = bindings["generation"].get_chat_template

        dialogs = eval(self.dialogs, bindings)
        pipe = pipeline(model="HuggingFaceTB/SmolLM2-360M-Instruct")

        template = get_chat_template()

        # sanity check
        if not isinstance(template, str):
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      template: "%s"' % template)
            grades.addMessage('\tThe chat template must be a string.')
            return False

        pipe.tokenizer.chat_template = template

        for dialog in dialogs:
            output = pipe(dialog, max_new_tokens=128)
            output = output[0]["generated_text"][-1]["content"]

            if self.is_refusal(output):
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\t%s' % self.failure)
                grades.addMessage('\t      question: "%s"' % dialog[-1]["content"])
                grades.addMessage('\tstudent result: "%s"' % output)
                return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\t%s' % self.success)

        return True
    
    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# File intentionally blank.\n')
        return True


class PreProcessingTest(testClasses.TestCase):

    def __init__(self, name, question, testDict):
        super(PreProcessingTest, self).__init__(name, question, testDict)
        self.actions = compile(testDict["actions"], "%s.test" % self.getPath(), "eval")
        self.success = testDict["success"]
        self.failure = testDict['failure']

    def cmp_list(self, l1, l2):
        if len(l1)!=len(l2):
            return False
        for i in range(len(l1)):
            if l1[i] != l2[i]:
                return False
        return True

    def execute(self, grades, moduleDict, solutionDict):
        # load the module
        bindings = dict(moduleDict)
        mapping_function = bindings["transformerGrammar"].mapping_function

        actions = eval(self.actions, bindings)

        inputs = eval(solutionDict["inputs"])
        labels = eval(solutionDict["labels"])
        position_ids = eval(solutionDict["position_ids"])
        attention_mask = eval(solutionDict["attention_mask"])

        for i, action in enumerate(actions):

            output = mapping_function({ "actions": action })
            if not isinstance(output, dict):
                grades.addMessage("FAIL: %s" % self.path)
                grades.addMessage("\t%s" % self.failure)
                grades.addMessage('\taction sequence: "{}"'.format(action))
                grades.addMessage('\t        student: "%s"' % output)
                grades.addMessage("\tThe output of the mapping function must be a dictionary.")
                return False
            
            if "inputs" not in output or "labels" not in output or "position_ids" not in output or "attention_mask" not in output:
                grades.addMessage("FAIL: %s" % self.path)
                grades.addMessage("\t%s" % self.failure)
                grades.addMessage('\taction sequence: "{}"'.format(action))
                grades.addMessage('\t        student: "%s"' % output)
                grades.addMessage("\tThe output of the mapping function must contain 'inputs', 'labels', 'position_ids' and 'attention_mask'.")
                return False
            
            if not isinstance(output["inputs"], list) or not isinstance(output["labels"], list) or not isinstance(output["position_ids"], list) or not isinstance(output["attention_mask"], torch.Tensor):
                grades.addMessage("FAIL: %s" % self.path)
                grades.addMessage("\t%s" % self.failure)
                grades.addMessage('\taction sequence: "{}"'.format(action))
                grades.addMessage('\t        student: "%s"' % output)
                grades.addMessage("\tThe output of the mapping function must contain 'inputs', 'labels', 'position_ids' and 'attention_mask' with proper type.")
                return False

            if not self.cmp_list(output["inputs"], inputs[i]):
                grades.addMessage("FAIL: %s" % self.path)
                grades.addMessage("\tWrong processed input.")
                grades.addMessage('\taction sequence: "{}"'.format(action))
                grades.addMessage('\t        student: "{}"'.format(output["inputs"]))
                grades.addMessage('\t        correct: "{}"'.format(inputs[i]))
                return False

            if not self.cmp_list(output["labels"], labels[i]):
                grades.addMessage("FAIL: %s" % self.path)
                grades.addMessage("\tWrong processed labels.")
                grades.addMessage('\taction sequence: "{}"'.format(action))
                grades.addMessage('\t        student: "{}"'.format(output["labels"]))
                grades.addMessage('\t        correct: "{}"'.format(labels[i]))
                return False

            if not self.cmp_list(output["position_ids"], position_ids[i]):
                grades.addMessage("FAIL: %s" % self.path)
                grades.addMessage("\tWrong position ids.")
                grades.addMessage('\taction sequence: "{}"'.format(action))
                grades.addMessage('\t        student: "{}"'.format(output["position_ids"]))
                grades.addMessage('\t        correct: "{}"'.format(position_ids[i]))
                return False

            att_mask = torch.tensor(attention_mask[i])
            if not att_mask.equal(output["attention_mask"]):
                grades.addMessage("FAIL: %s" % self.path)
                grades.addMessage("\tWrong attention mask.")
                grades.addMessage('\taction sequence: "{}"'.format(action))
                grades.addMessage('\t        student: "{}"'.format(output["attention_mask"]))
                grades.addMessage('\t        correct: "{}"'.format(att_mask))
                return False

        grades.addMessage("PASS: %s" % self.path)
        grades.addMessage("\t%s" % self.success)
        return True

    def writeSolution(self, moduleDict, filePath):
        bindings = dict(moduleDict)
        mapping_function = bindings["transformerGrammar"].mapping_function

        actions = eval(self.actions)

        results = [mapping_function({ "actions": action }) for action in actions]
        inputs = [res["inputs"] for res in results]
        labels = [res["labels"] for res in results]
        position_ids = [res["position_ids"] for res in results]
        attention_mask = [res["attention_mask"].numpy().tolist() for res in results]

        with open(filePath, "w") as handle:
            handle.write("# This is the solution file for %s.\n" % self.path)
            handle.write(
                "# The process results of each action sequence must equal the below.\n"
            )

            handle.write('inputs: """\n[\n')
            for i in inputs[:-1]:
                handle.write("    %s,\n" % str(i))
            handle.write("    %s\n" % str(inputs[-1]))
            handle.write(']\n"""\n')
            handle.write("\n")

            handle.write('labels: """\n[\n')
            for i in labels[:-1]:
                handle.write("    %s,\n" % str(i))
            handle.write("    %s\n" % str(labels[-1]))
            handle.write(']\n"""\n')
            handle.write("\n")

            handle.write('position_ids: """\n[\n')
            for i in position_ids[:-1]:
                handle.write("    %s,\n" % str(i))
            handle.write("    %s\n" % str(position_ids[-1]))
            handle.write(']\n"""\n')
            handle.write("\n")

            handle.write('attention_mask: """\n[\n')
            for i in attention_mask[:-1]:
                handle.write("    [%s,\n" % str(i[0]))
                for j in i[1:-1]:
                    handle.write("     %s,\n" % str(j))
                handle.write("     %s],\n" % str(i[-1]))
            handle.write("    [%s,\n" % str(attention_mask[-1][0]))
            for j in attention_mask[-1][1:-1]:
                handle.write("     %s,\n" % str(j))
            handle.write("     %s]\n" % str(attention_mask[-1][-1]))
            handle.write(']\n"""\n')
            handle.write("\n")
        return True


class TrainerTest(testClasses.TestCase):

    def __init__(self, name, question, testDict):
        super(TrainerTest, self).__init__(name, question, testDict)
        self.ds_kwargs = compile(testDict['ds_kwargs'], "%s.test" % self.getPath(), 'eval')
        self.num_samples = int(testDict['num_samples'])
        self.thresholds = [float(t) for t in testDict['thresholds'].split()]
        self.success = testDict['success']
        self.failure = testDict['failure']

        self.random = util.random.Random()
        self.random.seed(testDict['random_seed'])

    def execute(self, grades, moduleDict, solutionDict):
        # load the module
        bindings = dict(moduleDict)
        get_topic_classification_pipeline = bindings["classification"].get_topic_classification_pipeline

        pipe = get_topic_classification_pipeline()

        # sanity check
        sample = "Would the US constitution be changed if the admendment received 2/3 of the popular vote?"
        output = pipe(sample)
        if not isinstance(output, dict):
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % sample)
            grades.addMessage('\tstudent result: "%s"' % output)
            grades.addMessage('\tThe output of the pipeline must be a dictionary.')
            return False

        if "label" not in output or "score" not in output:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % sample)
            grades.addMessage('\tstudent result: "%s"' % output)
            grades.addMessage('\tThe output of the pipeline must contain "label" and "score".')
            return False
        
        if not isinstance(output["label"], str):
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % sample)
            grades.addMessage('\tstudent result: "%s"' % output)
            grades.addMessage('\tThe label of the output must be a string.')
            return False
        
        if not isinstance(output["score"], float):
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % sample)
            grades.addMessage('\tstudent result: "%s"' % output)
            grades.addMessage('\tThe score of the output must be a float.')
            return False
        
        if output["score"] < 0 or output["score"] > 1:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % sample)
            grades.addMessage('\tstudent result: "%s"' % output)
            grades.addMessage('\tThe score of the output must be in [0, 1].')
            return False
        
        if output["label"] not in ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % sample)
            grades.addMessage('\tstudent result: "%s"' % output)
            grades.addMessage('\tThe label of the output must be one of the following: "Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government".')
            return False
        
        # test the pipeline
        ds_kwargs = eval(self.ds_kwargs, bindings)
        ds = datasets.load_dataset(**ds_kwargs)
        ds = ds.select(self.random.sample(range(len(ds)), self.num_samples))
        labels = ds.features['topic'].names

        cnts = [0, 0]
        failed_sample = None
        for data in ds:
            text = data['question_title']
            result = pipe(text)

            if labels[data['topic']] != result["label"]:
                cnts[1] += 1
                failed_sample = (text, result["label"], labels[data['topic']])
            
            cnts[0] += 1

        accuracy = 1 - cnts[1] / cnts[0]

        points = 0
        for threshold in self.thresholds:
            if accuracy >= threshold:
                points += 1
        grades.addPoints(points)
        if points >= len(self.thresholds):
            grades.addMessage('PASS: %s' % self.path)
            grades.addMessage('\t%s' % self.success)
        else:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\t      sentence: "%s"' % failed_sample[0])
            grades.addMessage('\tstudent result: "%s"' % failed_sample[1])
            grades.addMessage('\tcorrect result: "%s"' % failed_sample[2])
        grades.addMessage('\taccuracy: %s' % round(accuracy, 2))
        grades.addMessage('\tthresholds: %s' % str(tuple(round(threshold, 2) for threshold in self.thresholds)))

        return True

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# File intentionally blank.\n')
        return True
