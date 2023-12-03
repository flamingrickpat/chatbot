summary_template = """
Summarize the conversation between this 2 people. Try to retain all relevant information. Always use their names instead of he or she!
 
Conversation: <conv>

Summary: """

emotion_template = """
The following table describes the emotions a person named <name> is feeling as value. Your goal is to summarize how <name> is currently feeling in a single sentence. Don't mention very weak emotions!
Emotions:
<emotions>

Current feeling of <name> in a single sentence: """
