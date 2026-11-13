from aster_vale.core.gender import predict_gender


def test_agent_address_female():
    t = "Agent: Hello ma'am.\nCustomer: Hello.\nAgent: Yes ma'am, tell me.\nCustomer: My knee hurts."
    assert predict_gender(t)["caller_gender"] == "female"


def test_customer_saying_sir_to_agent_is_ignored():
    t = "Customer: Sir, my back hurts.\nAgent: Okay ma'am.\nCustomer: Yes sir, yes sir.\nAgent: Thank you ma'am."
    assert predict_gender(t)["caller_gender"] == "female"


def test_self_kinship_and_patient_relation():
    t = "Agent: Hello.\nCustomer: I will consult my wife. It is for my mother, her knee.\nAgent: Okay sir."
    g = predict_gender(t)
    assert g["caller_gender"] == "male" and g["patient_gender"] == "female"


def test_spouse_caller_introduction():
    t = "Agent: Am I speaking with Renu ji?\nCustomer: Yes, I am Renu's husband speaking.\nAgent: Okay."
    g = predict_gender(t)
    assert g["caller_gender"] == "male" and g["patient_gender"] == "female"


def test_abstains_without_cues():
    assert predict_gender("Agent: Hello.\nCustomer: Hello.")["caller_gender"] == "unknown"
