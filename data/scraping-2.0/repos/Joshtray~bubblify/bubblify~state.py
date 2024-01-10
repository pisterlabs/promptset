"""Base state for the app."""

from langchain.chat_models import ChatOpenAI

import os

import reflex as rx

from collections import Counter

import random

from bubblify import styles

from .utils.auth import Auth

from bubblify.helpers.sql_helpers import (
    get_json_from_database,
    create_categories,
    insert_email_info,
    execute_sql_query,
    conn,
    insert_categorized_email,
    create_emails_info_table,
)
from quickstart import main

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))


class State(rx.State):
    """Base state for the app.

    The base state is used to store general vars used throughout the app.
    """

    email_data: list[dict] = [
        {
            "sender": "GitGuardian <security@getgitguardian.com>",
            "snippet": "GitGuardian has detected the following Google OAuth2 Keys exposed within your GitHub account. Details - Secret type: Google OAuth2 Keys - Repository: Joshtray/bubblify - Pushed date: October 29th 2023,",
            "subject": "[Joshtray/bubblify] Google OAuth2 Keys exposed on GitHub",
            "date_received": "2023-10-28",
            "unread": False,
            "category_name": "INBOX",
        },
        {
            "sender": "John Doe <john.doe@example.com>",
            "snippet": "Hello, just checking in on our project progress. How's it going?",
            "subject": "Project Progress",
            "date_received": "2023-10-27",
            "unread": True,
            "category_name": "INBOX",
        },
        {
            "sender": "Alice Smith <alice.smith@example.com>",
            "snippet": "Meeting reminder for next week. Don't forget to prepare the presentation.",
            "subject": "Meeting Reminder",
            "date_received": "2023-10-26",
            "unread": False,
            "category_name": "INBOX",
        },
        {
            "sender": "Support Team <support@example.com>",
            "snippet": "Your support ticket #12345 has been resolved. If you have any more questions, feel free to ask.",
            "subject": "Support Ticket Resolution",
            "date_received": "2023-10-25",
            "unread": False,
            "category_name": "INBOX",
        },
        {
            "sender": "Jane Williams <jane.williams@example.com>",
            "snippet": "Weekly report attached. Please review and provide feedback.",
            "subject": "Weekly Report",
            "date_received": "2023-10-24",
            "unread": True,
            "category_name": "INBOX",
        },
        {
            "sender": "Free Offers <offers@example.com>",
            "snippet": "Congratulations! You've won a free cruise vacation. Click here to claim your prize!",
            "subject": "Free Cruise Offer",
            "date_received": "2023-10-27",
            "unread": False,
            "category_name": "Spam",
        },
        {
            "sender": "Growth Hacks <hacks@example.com>",
            "snippet": "Get rich quick with our amazing investment opportunity. Don't miss out!",
            "subject": "Investment Opportunity",
            "date_received": "2023-10-26",
            "unread": False,
            "category_name": "Spam",
        },
        {
            "sender": "Unsolicited Newsletter <newsletter@example.com>",
            "snippet": "You're receiving this email because you subscribed to our newsletter. To unsubscribe, click here.",
            "subject": "Weekly Newsletter",
            "date_received": "2023-10-25",
            "unread": False,
            "category_name": "Spam",
        },
        {
            "sender": "Amazon Deals <deals@amazon.com>",
            "snippet": "Check out our latest deals and discounts on electronics, clothing, and more!",
            "subject": "Amazon Promotions",
            "date_received": "2023-10-28",
            "unread": False,
            "category_name": "Promotions",
        },
        {
            "sender": "Tech Store <info@techstore.com>",
            "snippet": "Exclusive offer for tech enthusiasts: 20% off on all gadgets this week only!",
            "subject": "Tech Store Promotion",
            "date_received": "2023-10-27",
            "unread": False,
            "category_name": "Promotions",
        },
        {
            "sender": "Fashion Outlet <sales@fashionoutlet.com>",
            "snippet": "New arrivals and special discounts on fashion items. Shop now!",
            "subject": "Fashion Outlet Sale",
            "date_received": "2023-10-26",
            "unread": False,
            "category_name": "Promotions",
        },
        {
            "sender": "Travel Discounts <info@traveldiscounts.com>",
            "snippet": "Plan your next vacation with our exclusive travel deals and discounts!",
            "subject": "Travel Discounts",
            "date_received": "2023-10-25",
            "unread": False,
            "category_name": "Promotions",
        },
        {
            "sender": "Food Delivery <offers@fooddelivery.com>",
            "snippet": "Get 10% off on your next food delivery order. Use code: DELICIOUS10",
            "subject": "Food Delivery Discount",
            "date_received": "2023-10-24",
            "unread": False,
            "category_name": "Promotions",
        },
    ]

    clusters: list[tuple[str, int, list, float, float, float]] = []
    index_index: int = 0
    name_index: int = 1
    size_index: int = 2
    messages_index: int = 3
    diameter_index: int = 4
    positionx_index: int = 5
    positiony_index: int = 6
    color_index: int = 7
    z_index_index: int = 8
    unread_count_index: int = 9

    colors: list[str] = [
        "#d27cbf",
        "#d2bf7c",
        "#7cb3d2",
        "#7cd2be",
        "#d27c7c",
        "#7cd2b3",
        "#d27cbf",
        "#7cbfd2",
    ]
    cluster_names: list[str] = ["Work", "Social", "Urgent", "Other"]
    new_cluster_name: str = ""
    current_email: str = ""
    current_password: str = ""
    authenticated_user: bool = False
    have_emails: bool = False
    email_data: list[dict] = [
        {
            "sender": "GitGuardian <security@getgitguardian.com>",
            "snippet": "GitGuardian has detected the following Google OAuth2 Keys exposed within your GitHub account. Details - Secret type: Google OAuth2 Keys - Repository: Joshtray/bubblify - Pushed date: October 29th 2023,",
            "subject": "[Joshtray/bubblify] Google OAuth2 Keys exposed on GitHub",
            "date_received": "2023-10-28",
            "unread": False,
            "category_name": "INBOX",
        },
        {
            "sender": "John Doe <john.doe@example.com>",
            "snippet": "Hello, just checking in on our project progress. How's it going?",
            "subject": "Project Progress",
            "date_received": "2023-10-27",
            "unread": True,
            "category_name": "INBOX",
        },
        {
            "sender": "Alice Smith <alice.smith@example.com>",
            "snippet": "Meeting reminder for next week. Don't forget to prepare the presentation.",
            "subject": "Meeting Reminder",
            "date_received": "2023-10-26",
            "unread": False,
            "category_name": "INBOX",
        },
        {
            "sender": "Support Team <support@example.com>",
            "snippet": "Your support ticket #12345 has been resolved. If you have any more questions, feel free to ask.",
            "subject": "Support Ticket Resolution",
            "date_received": "2023-10-25",
            "unread": False,
            "category_name": "INBOX",
        },
        {
            "sender": "Jane Williams <jane.williams@example.com>",
            "snippet": "Weekly report attached. Please review and provide feedback.",
            "subject": "Weekly Report",
            "date_received": "2023-10-24",
            "unread": True,
            "category_name": "INBOX",
        },
        {
            "sender": "Free Offers <offers@example.com>",
            "snippet": "Congratulations! You've won a free cruise vacation. Click here to claim your prize!",
            "subject": "Free Cruise Offer",
            "date_received": "2023-10-27",
            "unread": False,
            "category_name": "Spam",
        },
        {
            "sender": "Growth Hacks <hacks@example.com>",
            "snippet": "Get rich quick with our amazing investment opportunity. Don't miss out!",
            "subject": "Investment Opportunity",
            "date_received": "2023-10-26",
            "unread": False,
            "category_name": "Spam",
        },
        {
            "sender": "Unsolicited Newsletter <newsletter@example.com>",
            "snippet": "You're receiving this email because you subscribed to our newsletter. To unsubscribe, click here.",
            "subject": "Weekly Newsletter",
            "date_received": "2023-10-25",
            "unread": False,
            "category_name": "Spam",
        },
        {
            "sender": "Amazon Deals <deals@amazon.com>",
            "snippet": "Check out our latest deals and discounts on electronics, clothing, and more!",
            "subject": "Amazon Promotions",
            "date_received": "2023-10-28",
            "unread": False,
            "category_name": "Promotions",
        },
        {
            "sender": "Tech Store <info@techstore.com>",
            "snippet": "Exclusive offer for tech enthusiasts: 20% off on all gadgets this week only!",
            "subject": "Tech Store Promotion",
            "date_received": "2023-10-27",
            "unread": False,
            "category_name": "Promotions",
        },
        {
            "sender": "Fashion Outlet <sales@fashionoutlet.com>",
            "snippet": "New arrivals and special discounts on fashion items. Shop now!",
            "subject": "Fashion Outlet Sale",
            "date_received": "2023-10-26",
            "unread": False,
            "category_name": "Promotions",
        },
        {
            "sender": "Travel Discounts <info@traveldiscounts.com>",
            "snippet": "Plan your next vacation with our exclusive travel deals and discounts!",
            "subject": "Travel Discounts",
            "date_received": "2023-10-25",
            "unread": False,
            "category_name": "Promotions",
        },
        {
            "sender": "Food Delivery <offers@fooddelivery.com>",
            "snippet": "Get 10% off on your next food delivery order. Use code: DELICIOUS10",
            "subject": "Food Delivery Discount",
            "date_received": "2023-10-24",
            "unread": False,
            "category_name": "Promotions",
        },
    ]

    prev_index: int = 0
    prev_diameter: float = 0
    prev_pos_x: float = 0
    prev_pos_y: float = 0
    in_focus: bool = False
    messages: list[list[tuple[str]]] = []

    def get_clusters(self):
        """Get the clusters from the database.

        Returns:
            The clusters.
        """

        clusters = {i: [] for i in self.cluster_names}
        for message in self.email_data:
            if message["category_name"] not in clusters:
                clusters[message["category_name"]] = []
            clusters[message["category_name"]].append(message)

        for category in clusters:
            msgs = []
            for msg in clusters[category]:
                msgs.append((msg["snippet"], msg["sender"], msg["date_received"], msg["subject"], msg["unread"]))
            
            self.messages.append(msgs)

        print(self.messages)


        output = self.messages
        diameters = self.get_diameters(clusters)
        positions = self.get_positions(clusters, diameters)
        colors = self.get_colors(clusters)
        unread_counts = self.get_unread_count(clusters)
        self.clusters = [
            (
                i,
                name,
                len(clusters[name]),
                clusters[name],
                diameters[i],
                positions[i][0],
                positions[i][1],
                colors[i],
                1,
                unread_counts[i],
            )
            for i, name in enumerate(clusters)
        ]

    def get_diameters(self, clusters):
        """Get the diameters of the clusters.

        Returns:
            The diameters.
        """
        n = len(clusters)
        min_size = styles.min_bubble_size
        max_size = 1200 / n
        mean = sum([len(clusters[cluster]) for cluster in clusters]) / len(clusters)
        deviations = [(len(clusters[cluster]) - mean) / mean for cluster in clusters]
        diff = (max_size - min_size) / 2
        avg_diam = (max_size + min_size) / 2
        return [((dev * diff) + avg_diam) for dev in deviations]

    def get_positions(self, clusters, diameters):
        """Get the positions of the clusters.

        Returns:
            The positions.
        """
        positions = []
        n = len(clusters)
        x_ranges = [
            (-600 + i * (1200 / n), -600 + (i + 1) * (1200 / n)) for i in range(n)
        ]

        for i in range(n):
            diameter = diameters[i]

            x_range_ind = random.choice(range(len(x_ranges)))

            x = random.uniform(
                x_ranges[x_range_ind][0], x_ranges[x_range_ind][1] - diameter
            )

            x_ranges.pop(x_range_ind)

            y = random.uniform(-(800 - diameter) / 2, (800 - diameter) / 2)

            positions.append((str(x) + "px", str(y - (diameter / 2)) + "px"))
        return positions

    def get_colors(self, clusters):
        """Get the colors of the clusters.

        Returns:
            The colors.
        """
        return random.sample(self.colors, k=len(clusters))

    def get_unread_count(self, clusters):
        """Get the unread count of the cluster.

        Returns:
            The unread count.
        """
        return [
            len([message for message in clusters[cluster] if message["unread"]])
            for cluster in clusters
        ]

    def mouse_enter(self, cluster):
        """Mouse enter the bubble.

        Returns:
            The new bubble size.
        """
        if not self.in_focus:
            new_cluster = list(cluster)
            new_cluster[self.diameter_index] = cluster[self.diameter_index] * 1.1

            self.clusters[cluster[self.index_index]] = tuple(new_cluster)

    def mouse_leave(self, cluster):
        """Mouse leave the bubble.

        Returns:
            The new bubble size.
        """
        if not self.in_focus:
            new_cluster = list(cluster)
            new_cluster[self.diameter_index] = cluster[self.diameter_index] / 1.1

            self.clusters[cluster[self.index_index]] = tuple(new_cluster)

    def bubble_click(self, cluster):
        """Click the bubble.

        Returns:
            The new bubble size.
        """
        if not self.in_focus:
            self.in_focus = True
            self.prev_index = cluster[self.index_index]
            self.prev_diameter = cluster[self.diameter_index]
            self.prev_pos_x = cluster[self.positionx_index]
            self.prev_pos_y = cluster[self.positiony_index]

            new_cluster = list(cluster)
            new_cluster[self.diameter_index] = "90vh"
            new_cluster[self.positionx_index] = "-45vh"
            new_cluster[self.positiony_index] = "-45vh"
            new_cluster[self.z_index_index] = 3

            self.clusters[cluster[self.index_index]] = tuple(new_cluster)

    def bubble_close(self):
        """Close the bubble.

        Returns:
            The new bubble size.
        """
        if not self.in_focus:
            return

        self.in_focus = False
        cluster = self.clusters[self.prev_index]
        new_cluster = list(cluster)
        new_cluster[self.diameter_index] = self.prev_diameter
        new_cluster[self.positionx_index] = self.prev_pos_x
        new_cluster[self.positiony_index] = self.prev_pos_y
        new_cluster[self.z_index_index] = 1

        self.clusters[cluster[self.index_index]] = tuple(new_cluster)

    def add_cluster(self, cluster_name):
        """
        Args:
            cluster_name: The name of the cluster to add.
        """
        if cluster_name in self.cluster_names or cluster_name == "":
            rx.alert("Please enter a unique cluster name")
        else:
            self.cluster_names.append(cluster_name)

    def delete_cluster(self, cluster_name):
        """
        Args:
            cluster_name: The name of the cluster to delete.
        """
        for i in range(len(self.clusters) - 1, -1, -1):
            if self.cluster_names[i][self.name_index] == cluster_name:
                self.cluster_names.pop(i)

    def set_new_cluster_name(self, new_cluster_name):
        """
        Args:
            new_cluster_name: The new cluster name.
        """
        self.new_cluster_name = new_cluster_name

    def login(self, email, password):
        """
        Args:
            email: The email.
            password: The password.
        """

        if Auth.get_user(email):
            if Auth.authenticate_user(email, password):
                self.authenticated_user = True
            else:
                rx.alert("Incorrect password")
        else:
            print("user does not exist")
            if email != "" and password != "":
                Auth.create_new_user(email, password)
                self.authenticated_user = True
                print("This happened")
                rx.redirect("/", True)

    def logout(self):
        self.authenticated_user = False

    def connect_google(self):
        if os.path.exists("token.json"):
            os.remove("token.json")
        create_emails_info_table()
        main()
        self.email_data = get_json_from_database()
        self.have_emails = True
        self.categorize()

    def categorize(self):
        # Doing this in chunks of 10 emails at a time

        for i in range(0, len(self.email_data), 10):
            prompt = f"""
            For the following emails, please categorize them as one of the following:
            {self.cluster_names}
            
            Emails: {self.email_data[i:i+10]}\n
            
            Return only the category name in the following format (each separated by a new line)):
            
            <category_name>
            <category_name>
            <category_name>
            """

            response = llm.predict(prompt)
            response = response.split("\n")
            for j in range(len(response)):
                self.email_data[i + j]["category_name"] = response[j]

        # for i, email in enumerate(self.email_data):
        #     prompt = f"""
        #     For the following email, please categorize it as one of the following:
        #     {self.cluster_names}

        #     Email: {email['snippet']}\n
        #     Sender: {email['sender']}\n
        #     Subject: {email['subject']}\n
        #     Date: {email['date_received']}\n
        #     Unread: {email['unread']}\n

        #     Return only the category name in the following format:

        #     <category_name>
        #     """

        #     response = llm.predict(prompt)
        #     self.email_data[i]["category_name"] = response

        self.get_clusters()
