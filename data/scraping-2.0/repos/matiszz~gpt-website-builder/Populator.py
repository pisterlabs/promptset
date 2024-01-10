from OpenAIController import OpenAIController


def get_pricing_feature(text):
    html = """
    <p class="flex items-center text-gray-600 mb-2">
      <span class="w-4 h-4 mr-2 inline-flex items-center justify-center bg-gray-400 text-white rounded-full flex-shrink-0">
        <svg fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5"
             class="w-3 h-3" viewBox="0 0 24 24">
          <path d="M20 6L9 17l-5-5"></path>
        </svg>
      </span>{feature}
    </p>
    """
    return html.format(feature=text)


class Populator(object):
    openAI = OpenAIController()

    def __init__(self):
        pass

    @staticmethod
    def populate_contact_html(content, address, email, phone_number):
        return content.format(address=address, email=email, phone_number=phone_number, company_location=address.replace(" ", "+"))

    def populate_features_html(self, content, web_info):
        features_1 = self.openAI.get_feature(web_info)
        features_2 = self.openAI.get_feature(web_info)
        features_3 = self.openAI.get_feature(web_info)

        return content.format(
            web_name=web_info['web_name'],
            feature_1_name=features_1[0],
            feature_1_description=features_1[1],
            feature_2_name=features_2[0],
            feature_2_description=features_2[1],
            feature_3_name=features_3[0],
            feature_3_description=features_3[1],
            image_src_features=web_info['photo2']
        )

    @staticmethod
    def populate_footer_html(content, web_name):
        return content.format(web_name=web_name)

    @staticmethod
    def populate_head_html(content, web_name):
        return content.format(web_name=web_name)

    def populate_testimonial_html(self, content, web_info):
        features = self.openAI.get_testimonial_features(web_info)
        return content.format(
            testimonial_1=features['testimonials'][0],
            testimonial_2=features['testimonials'][1],
            testimonial_3=features['testimonials'][2],
            name_1=features['names'][0],
            name_2=features['names'][1],
            name_3=features['names'][2],
            role_1=features['roles'][0],
            role_2=features['roles'][1],
            role_3=features['roles'][2],
        )

    def populate_hero_html(self, content, description, image_src):
        tagline = self.openAI.get_tagline(description)
        copy = self.openAI.get_copy(description)

        return content.format(tagline=tagline, copy=copy, image_src=image_src)

    def populate_navbar_html(self, content, web_name, description):
        links = self.openAI.get_navbar_links(description)
        return content.format(web_name=web_name, link_1=links[0], link_2=links[1], link_3_cta=links[2])

    def populate_pricing_html(self, content, description, web_name):
        features = self.openAI.get_pricing_features(description)
        features_start = ""
        features_pro = ""
        features_business = ""

        for feature in features['start']:
            if feature != '':
                features_start += get_pricing_feature(feature)
        for feature in features['pro']:
            if feature != '':
                features_pro += get_pricing_feature(feature)
        for feature in features['business']:
            if feature != '':
                features_business += get_pricing_feature(feature)

        return content.format(
            web_name=web_name,
            features_start=features_start,
            features_pro=features_pro,
            features_business=features_business
        )
