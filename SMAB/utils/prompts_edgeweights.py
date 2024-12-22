# Prompts for calculating edge weights using LLM

prompt1 = ''' Given a sentence, rate whether the sentence appears to be an example  from a hate speech detection dataset (either with hate/non-hate label), from 1 (lowest) - 4 (highest). We call this score: In-distribution rating.
    Use the following binary word-level definitions: 
    (i) 'Polarity of word (NEGpol)': Whether a word is a negative word(e.g. kill) or a positive word(e.g. happy)
    (ii) 'Intensity of negative word (NEGint): Whether it is a weak intensity negative  word (e.g. dirty) or a strong intensity negative word (e.g. filthy).
    (iii) 'Negative Word Directed towards a Person (NEGperson)': Whether it is a negative word directed towards a person in the sentence.

The detailed criterion for rating is as follows: 4(highest): The sentence contains at least two words with negative polarity(NEGpol), 3: The sentence contains 1 negative word(NEGpol) which has strong intensity(NEGint) OR a negative word specifically Directed towards a Person (NEGperson), 2: The sentence contains 1 negative word(NEGpol) which has low intensity, 1(lowest): The sentence is barely relevant to the task of hate classification or completely irrelevant.
--------------------------------------
Sentence: Tokyo is a vibrant city full of exciting tourist attractions.
In-distribution rating: 1
Explanation:The sentence "Tokyo is a vibrant city full of exciting tourist attractions" does not contain any negative words (NEGpol). There are no negative words directed towards persons (NEGperson), no words with strong intensity (NEGint). The sentence is positive in nature and is not relevant to the task of hate classification. Therefore, it receives the lowest in-distribution rating.

Sentence: if you think that islam is racist you must think islam is a race.
In-distribution rating: 3
Explanation: The sentence "if you think that islam is racist you must think islam is a race. and if you think islam is a race you are a..." contains a negative word "racist" (NEGpol). And the negative word has strong intensity (NEGint). Given that there is one negative word with strong intensity, it aligns with the criterion for a rating of 3, as it is relevant to hate classification but does not meet the stronger criteria for 4 rating.

Sentence: kill isis. kill kill. love jesus.
In-distribution rating: 4
Explanation: "kill isis. kill kill. love america. love jesus." contains multiple negative words "kill" (NEGpol) repeated several times, which intensifies their impact. The word "kill" is a strong intensity negative word (NEGint). Additionally, "isis" refers to a specific group, implying the negative word is directed towards a person or group (NEGperson). Given the presence of multiple negative polarity words and their strong intensity, this sentence aligns with the criteria for the highest rating, 4, indicating it is highly relevant to hate classification.

Sentence: yeah ryan. but you don\'t mind " clogging up " " jails with american jail...
In-distribution rating: 2
Explanation: The sentence "yeah ryan. but you don't mind 'clogging up' jails with american jail..." contains the negative phrase "clogging up" (NEGpol). While this phrase is negative, it has weak intensity (NEGint). The phrase does not specifically direct negativity towards a person (NEGperson). Given that there is one negative word with low intensity, it aligns with the criterion for a rating of 2, indicating some relevance to hate classification but not strongly so.
'''





prompt2 = '''Given a sentence, rate whether the sentence appears to be an example from a hate speech detection dataset (either with hate/non-hate label), from 1 (lowest) - 5 (highest). We call this score: In-distribution rating.
    Utilize the following binary word-level definitions: 
    (i) 'Polarity of word (NEGpol)': Whether a word is a negative word(e.g. kill) or a positive word(e.g. happy)
    (ii) 'Intensity of negative word (NEGint): Whether it is a weak intensity negative  word (e.g. dirty) or a strong intensity negative word (e.g. filthy).
    (iii) 'Negative Word Directed towards a Person/Group (NEGperson)': Whether it is a negative word directed towards a person or a group in the sentence.
  
The criterion for rating is as follows: 5(highest): The sentence is VERY LIKELY to be related the task of hate classification, 4: The sentence is LIKELY to be related the task of hate classification, 3: The sentence may or may not be related the task of hate classification it is NEUTRAL, 2: The sentence is UNLIKELY to be related the task of hate classification , 1(lowest): The sentence is VERY UNLIKELY to be related the task of hate classification.
--------------------------------------
Sentence: kill isis. kill kill. love jesus.
In-distribution rating: 5
Explanation: The sentence "kill isis. kill kill. love jesus." contains multiple negative words "kill" (NEGpol) repeated several times, which intensifies their impact. The word "kill" is a strong intensity negative word (NEGint). Additionally, "isis" refers to a specific group, implying the negative word is directed towards a person or group (NEGperson). Given the presence of multiple negative polarity words, their strong intensity, and the clear direction towards a specific group, this sentence is very likely to be related to the task of hate classification, thus it aligns with the highest rating, 5.

Sentence: if you think that islam is racist you must think islam is a race.
In-distribution rating: 4
Explanation: The sentence "if you think that islam is racist you must think islam is a race" contains a negative word "racist" (NEGpol) which has strong intensity (NEGint). Additionally, the term "racist" can be seen as indirectly directed towards individuals or groups (NEGperson) who are accused of being racist. Given the presence of a strong intensity negative word and its relevance to hate speech classification, this sentence is likely to be related to the task of hate classification, thus it aligns with a rating of 4.

Sentence: yeah ryan. but you don't mind " clogging up " " jails with american jail...
In-distribution rating: 2
Explanation: The sentence "yeah ryan. but you don't mind 'clogging up' 'jails with american jail..." contains no clear negative words (NEGpol) nor words with strong negative intensity (NEGint). While there is a mention of "jails" and "clogging up," these terms are not inherently negative or directed towards a person or group (NEGperson) in a way that strongly indicates hate speech. The context of the sentence is ambiguous and does not strongly relate to hate classification, making it unlikely to be related to the task. Therefore, it receives an in-distribution rating of 2.

Sentence: Tokyo is a vibrant city full of exciting tourist attractions.
In-distribution rating: 1
Explanation: The sentence "Tokyo is a vibrant city full of exciting tourist attractions" does not contain any negative words (NEGpol). There are no negative words directed towards persons or groups (NEGperson), and no words with strong intensity (NEGint). The sentence is positive in nature and is not relevant to the task of hate classification. Therefore, it receives the lowest in-distribution rating, 1.

Sentence: He calls himself a Zionist in the new interview.
In-distribution rating: 3
Explanation: The sentence "He calls himself a Zionist in the new interview." does not contain any explicit negative words (NEGpol) or strong intensity negative words (NEGint). However, the term "Zionist" can be politically and emotionally charged and may be perceived negatively by some groups, though it is not inherently a negative term. Since it is context-dependent and could potentially be related to discussions of hate speech or political identity, it is neutral in this context. Therefore, it receives an in-distribution rating of 3.
'''




prompt3 = '''
Given a sentence, rate whether the sentence appears to be an example from a hate speech detection dataset (either with hate/non-hate label), from 1 (lowest) - 5 (highest). We call this score: In-distribution rating.

The criterion for rating is as follows: 5(highest): The sentence is VERY LIKELY to be related the task of hate classification, 4: The sentence is LIKELY to be related the task of hate classification, 3: The sentence may or may not be related the task of hate classification it is NEUTRAL, 2: The sentence is UNLIKELY to be related the task of hate classification , 1(lowest): The sentence is VERY UNLIKELY to be related the task of hate classification.
--------------------------------------
Sentence: kill isis. kill kill. love jesus.
In-distribution rating: 5
Explanation: The sentence "kill isis. kill kill. love jesus." contains multiple negative words "kill" repeated several times, which intensifies their impact. The word "kill" is a strong intensity negative word . Additionally, "isis" refers to a specific group, implying the negative word is directed towards a person or group. Given the presence of multiple negative polarity words, their strong intensity, and the clear direction towards a specific group, this sentence is very likely to be related to the task of hate classification, thus it aligns with the highest rating, 5.

Sentence: if you think that islam is racist you must think islam is a race.
In-distribution rating: 4
Explanation: The sentence "if you think that islam is racist you must think islam is a race" contains a negative word "racist" which has strong intensity. Additionally, the term "racist" can be seen as indirectly directed towards individuals or groups who are accused of being racist. Given the presence of a strong intensity negative word and its relevance to hate speech classification, this sentence is likely to be related to the task of hate classification, thus it aligns with a rating of 4.

Sentence: yeah ryan. but you don't mind " clogging up " " jails with american jail...
In-distribution rating: 2
Explanation: The sentence "yeah ryan. but you don't mind 'clogging up' 'jails with american jail..." contains no clear negative words nor words with strong negative intensity . While there is a mention of "jails" and "clogging up," these terms are not inherently negative or directed towards a person or group  in a way that strongly indicates hate speech. The context of the sentence is ambiguous and does not strongly relate to hate classification, making it unlikely to be related to the task. Therefore, it receives an in-distribution rating of 2.

Sentence: Tokyo is a vibrant city full of exciting tourist attractions.
In-distribution rating: 1
Explanation: The sentence "Tokyo is a vibrant city full of exciting tourist attractions" does not contain any negative words . There are no negative words directed towards persons or groups , and no words with strong intensity . The sentence is positive in nature and is not relevant to the task of hate classification. Therefore, it receives the lowest in-distribution rating, 1.

Sentence: He calls himself a Zionist in the new interview.
In-distribution rating: 3
Explanation: The sentence "He calls himself a Zionist in the new interview." does not contain any explicit negative words  or strong intensity negative words . However, the term "Zionist" can be politically and emotionally charged and may be perceived negatively by some groups, though it is not inherently a negative term. Since it is context-dependent and could potentially be related to discussions of hate speech or political identity, it is neutral in this context. Therefore, it receives an in-distribution rating of 3.

'''
