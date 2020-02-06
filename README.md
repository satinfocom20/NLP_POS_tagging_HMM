# NLP_POS_tagging_HMM
Natural language processing (NLP) is an important research area in artificial intelligence, dating back to at least the 1950's. One of the most basic problems in NLP is part-of-speech tagging, in which the goal is to mark every word in a sentence with its part of speech (noun, verb, adjective,etc.). This is a first step towards extracting semantics from natural language text. For example, consider the following sentence:

Her position covers a number of daily tasks common to any social director.

Part-of-speech tagging here is not easy because many of these words can take on dierent parts of speech depending on context. For example, position can be a noun (as in the above sentence) or a verb (as in "They position themselves near the exit"). In fact, covers, number, and tasks can all be used as either nouns or verbs, while social and common can be nouns or adjectives, and daily can be an adjective, noun, or adverb. The correct labeling for the above sentence is:

Her position covers a   number of  daily tasks common to  any social director.
DET NOUN     VERB   DET NOUN   ADP ADJ   NOUN  ADJ    ADP DET ADJ    NOUN

where DET stands for a determiner, ADP is an adposition, ADJ is an adjective, and ADV is anadverb. Labeling parts of speech thus involves an understanding of the intended meaning of the words in the sentence, as well as the relationships between the words.

Build below 3 models
(a) HMM
S1-->W1; S1-->S2
S2-->W2; S2-->S3
S3-->W3; S3-->S4
...
SN-->Wn

(b) Simplified model
S1-->W1
S2-->W2
S3-->W3
S4-->W4
...
SN-->WN

(c) Complicated model
S1-->W1; S1-->S2; S1-->S3
S2-->W2; S2-->S3; S2-->S4
S3-->W3; S3-->S4; S3-->S5
...
SN-->WN
