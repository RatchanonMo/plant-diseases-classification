# Include PIL, load_image before main()
from pyrsistent import s
import streamlit as st
import os
from PIL import Image
from fastai.vision.all import (
    load_learner,
    PILImage,
)
import pathlib

MAIN_MODEL = pathlib.Path("./models/vgg-90.pkl")
GOOD_OR_BAD = pathlib.Path("./models/good_or_bad.pkl")

learn_inf = load_learner(MAIN_MODEL)
good_or_bad = load_learner(GOOD_OR_BAD)


def load_image(image_file):
    img = Image.open(image_file)
    return img


st.set_page_config(
    page_title="🍃 Plant Diseases Classification",
    layout="centered",

)




st.title("🍃 Plant Diseases Classification")
st.subheader("Something's wrong on your leaf? Let's find it.")

image_file = st.file_uploader("Upload Leaf Images",
                                  type=["png", "jpg", "jpeg"])

if image_file is not None:

    st.image(load_image(image_file), width=None)

    # Saving upload
    with open(os.path.join("./images/upload/", image_file.name), "wb") as f:
        f.write((image_file).getbuffer())

    
        file = f"./images/upload/{image_file.name}"

        # checkleaf = good_or_bad.predict(file)
        # if checkleaf[0] == 'good' :
        #     result = learn_inf.predict(file)
        #     st.success(result[0])
        # else:
        #     st.error('Bad image! Try again better image.')

        result = learn_inf.predict(file)

      

        predict = f"<div style='background:#00d26a;padding:0 20px 0 20px;border-radius:10px 10px 0 0;'><h1 style='color:#fff'>{result[0]}</h1></div>"
        st.markdown(predict, unsafe_allow_html=True)

        diseases = [
            "Black_rot",
            "Downy_mildew",
            "Tar_spot",
            "Healthy_leaf",
            "Powdery_mildew",
            "Sooty_mold",
            "Rust_leaf"
        ]

        description = [
            "<div style='background:none;padding:30px;color:#1c1d1d;border-radius:0 0 10px 10px;border:2px solid #00d26a'><b>Problem :</b><br/>It is a disease that affects almost all orchid species. especially the orchids of the Wanda group Caused by Phytophthora palmivora, it spreads easily during the rainy season, in high humidity climates, fungal spores spread. Distributed with the water used to water the plants. Should adjust the condition of the house to be transparent Distance allows the wind to blow through easily.<br/><br/><b>Solution :</b><br/>1.) Burn and destroy diseased plants. <br/>2.) Cut off the uninfected parts. The scissors used for cutting should be sterilized by flame or dipped in alcohol. When cut, it is painted with red cement to prevent germs from entering. <br/>3.) Use anti-fungal medication Once a week is a mencozeb such as manseb200, ditanM45 <br/>4.) The drug should be sprayed in the evening after sunset until dusk.</div>",
            "<div style='background:none;padding:30px;color:#1c1d1d;border-radius:0 0 10px 10px;border:2px solid #00d26a'><b>Problem :</b><br/> Downy mildew is caused by fungus-like organisms and affects many ornamentals and edibles, such as impatiens, pansies, columbine, grapevines, lettuce and cole crops such as broccoli and cauliflower. Often occurring during wet weather, downy mildew causes the upper portion of leaves to discolor, while the bottoms develop white or gray mold.<br/><br/><b>Solution :</b><br/> Plant resistant cultivars when available. No fungicides are available, but cultural practices can help. Remove and destroy infected foliage, or entire plants if downy mildew is prevalent. Avoid crowding plants or watering them in the evening, and rotate edibles year to year.</div>",
            "<div style='background:none;padding:30px;color:#1c1d1d;border-radius:0 0 10px 10px;border:2px solid #00d26a'><b>Problem :</b><br/>Alternaria leaf spot is one of the most prevalent fungal infestations. cause disease to plants Many species, especially vegetables, the symptoms of the disease can occur with all parts of the vegetable plant. and at all stages of growth from the embryo that begins to germinate from the seed until it grows into an old tree On seedlings, the first symptom is manifested as small, black-leaf wounds, similar to damping-off, on the stems. Vegetables that have been infected at this seedling stage will either stop growing or stall. If they are transplanted, they will become incomplete, slow growing plants. incomplete<br/><br/><b>Solution : </b><br/>",
            "<div style='background:none;padding:30px;color:#1c1d1d;border-radius:0 0 10px 10px;border:2px solid #00d26a'>Great job! Keep taking good care of it.</div>",
            "<div style='background:none;padding:30px;color:#1c1d1d;border-radius:0 0 10px 10px;border:2px solid #00d26a'><b>Problem : </b><br/>Powdery mildew leaves a telltale white dusty coating on leaves, stems and flowers. Caused by a fungus, it affects a number of plants, including lilacs, apples, grapes, cucumbers, peas, phlox, daisies and roses. <br/><br/><b>Solution :</b><br/> Rake up and destroy infected leaves to reduce the spread of spores. Also, give plants good drainage and ample air circulation. Avoid overhead watering at night; mid-morning is preferred to allow foliage to dry before evening. Commercial fungicides are available for powdery mildew, or you can spray with a solution of one tsp. baking soda and one quart of water as recommended by George “Doc” and Katy Abraham, authors of The Green Thumb Garden Handbook.</div>",
            "<div style='background:none;padding:30px;color:#1c1d1d;border-radius:0 0 10px 10px;border:2px solid #00d26a'><b>Problem : </b><br/>Sooty mold refers to fungi that grow on the sticky deposits, called honeydew, left by plant-sucking insects. On leaves it’s not only unsightly; it impedes photosynthesis and stunts plant growth. Leaves coated with sooty mold also drop off prematurely. <br/><br/><b>Solution :</b><br/> To deal with sooty mold, you have to deal with the plant-sucking insects leaving the honeydew, such as aphids, leafhoppers and mealybugs. Spray them with insecticidal soap or neem oil. Ants are attracted to the honeydew for food, so they protect the plant-sucking insects from predatory insects that would otherwise keep a lid on the pest population. Trap ants or, with woody plants, paint a sticky compound such as Tanglefoot around stems.</div>",
            "<div style='background:none;padding:30px;color:#1c1d1d;border-radius:0 0 10px 10px;border:2px solid #00d26a'><b>Problem : </b><br/>Rust, another fungal disease, is easy to spot because it forms rusty spots on leaves and sometimes stems. The spots eventually progress from reddish-orange to black. There are many types of rust that can attack plants such as hollyhocks, roses, daylilies and tomatoes. Even your lawn is susceptible to grass rust.  <br/><br/><b>Solution : </b><br/>Fungicides are available. Culturally, it’s a good practice to gather and destroy any infected plants to prevent the fungus from overwintering.</div>"
        ]

        index = diseases.index(result[0])

        st.markdown(description[index], unsafe_allow_html=True)

