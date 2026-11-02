
DISEASE_INFO = {

    'Acne and Rosacea Photos': {
            'causes': 'Hormonal changes, bacteria, genetics, stress, diet, and certain medications can contribute to acne and rosacea.',
            'precautions': 'Maintain a gentle skincare routine, avoid harsh products, manage stress, and consult a dermatologist for appropriate treatment.'
        },
        'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions': {
            'causes': 'Prolonged sun exposure, fair skin, age, and a history of sunburns can increase the risk of actinic keratosis and skin cancer.',
            'precautions': 'Use sunscreen, wear protective clothing, avoid tanning beds, and regularly check your skin for any changes or new growths.'
        },
        'Atopic Dermatitis Photos': {
            'causes': 'Genetics, environmental factors, and immune system dysfunction can contribute to atopic dermatitis.',
            'precautions': 'Maintain a consistent skincare routine, avoid known triggers, and consult a dermatologist for appropriate treatment.'
        },
        'Bullous Disease Photos': {
            'causes': 'Autoimmune disorders, genetic factors, and certain medications can lead to bullous diseases.',
            'precautions': 'Seek medical attention for proper diagnosis and treatment, avoid trauma to the skin, and follow prescribed medications and care routines.'  
        },
        'Cellulitis Impetigo and other Bacterial Infections': {
            'causes': 'Bacterial infections, often caused by Staphylococcus or Streptococcus bacteria, can lead to cellulitis and impetigo.',
            'precautions': 'Keep wounds clean and covered, practice good hygiene, and seek medical attention for proper diagnosis and treatment.'
        },
        'Eczema Photos': {
            'causes': 'Genetics, environmental factors, and immune system dysfunction can contribute to eczema.',
            'precautions': 'Maintain a consistent skincare routine, avoid known triggers, and consult a dermatologist for appropriate treatment.'
        },
        'Seborrheic Dermatitis Photos': {
            'causes': 'Genetics, hormonal changes, and the presence of a yeast-like fungus can contribute to seborrheic dermatitis.',
            'precautions': 'Use gentle, fragrance-free products, manage stress, and consult a dermatologist for appropriate treatment.'
        },
        'Exanthems and Drug Eruptions': {
            'causes': 'Viral infections, medications, and allergic reactions can lead to exanthems and drug eruptions.',
            'precautions': 'Identify and avoid triggers, follow prescribed medications, and seek medical attention for proper diagnosis and treatment.'         
        },
        'Hair Loss Photos Alopecia and other Hair Diseases': {
            'causes': 'Genetics, hormonal changes, autoimmune disorders, and certain medications can contribute to hair loss and alopecia.',
            'precautions': 'Maintain a healthy diet, manage stress, avoid harsh hair treatments,    and consult a dermatologist for appropriate treatment.'     
        },
        'Herpes HPV and other STDs Photos': {
            'causes': 'Sexually transmitted infections, including herpes and HPV, can lead to various skin conditions.',
            'precautions': 'Practice safe sex, get vaccinated for HPV, and seek medical attention for proper diagnosis and treatment of any suspected infections.'  
        },
        
        "Vascular Tumors": {
            "causes": (
                "Vascular tumors develop from blood vessel cells. Some are benign, such as hemangiomas, while others may be more serious. Their causes vary depending on the type of tumor."
            ),
            "precautions": (
                "Consult a dermatologist to identify the type of growth. Seek medical evaluation if it grows rapidly, bleeds repeatedly,changes in appearance, or becomes painful. Avoid attempting to remove the growth yourself."
            )
        },
    
        "Vasculities Photos": {
            "causes": (
                "Vasculitis is inflammation of blood vessels. It may be associated with autoimmune conditions, infections, certain medications, or other immune system reactions."
            ),
            "precautions": (
                "Seek medical evaluation for unexplained purple spots, skin ulcers,or persistent rashes. Obtain urgent medical attention if skin changes occur with severe pain, fever, blood in the urine, breathing difficulties, or other significant symptoms."
            )
        },
    
        "Warts Molluscum and other Viral Infections": {
            "causes": (
                "This category includes viral skin conditions caused by different viruses. Warts are commonly caused by human papillomavirus (HPV), while molluscum contagiosum is caused by a different virus.Some infections can spread through direct contact."
            ),
            "precautions": (
                "Avoid scratching or picking lesions, wash your hands regularly,and avoid sharing personal items that contact affected skin.Seek medical advice for persistent, spreading, painful, or uncertain lesions, particularly if you have a weakened immune system."
            )
        },

}


def get_disease_info(disease_name):

    # Case-insensitive matching
    for name, information in DISEASE_INFO.items():

        if name.lower() == disease_name.lower():
            return information

    # Fallback for classes without verified information
    return {
        "description": (
            "Educational information for this predicted class "
            "has not yet been added."
        ),
        "causes": [
            "The model cannot determine the actual cause of a skin condition."
        ],
        "precautions": [
            "Avoid scratching or irritating the affected area.",
            "Keep the skin clean using gentle products.",
            "Do not self-medicate based on an AI prediction.",
            "Consult a qualified dermatologist for assessment."
        ],
        "consult": (
            "A dermatologist should evaluate the skin condition "
            "before any treatment is started."
        )
    }
