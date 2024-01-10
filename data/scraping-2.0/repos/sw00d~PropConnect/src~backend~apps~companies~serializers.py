# serializers.py
from openai.error import InvalidRequestError
from rest_framework import serializers

from settings.base import STRIPE_SECRET_KEY
from .models import Company

from djstripe.models import Customer, PaymentMethod, Subscription
import stripe

stripe.api_key = STRIPE_SECRET_KEY


class PaymentMethodSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaymentMethod
        fields = ['billing_details']


class CompanyCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Company
        fields = (
            'id',
            'name',
            'website',
            'number_of_doors',
            'zip_code',
        )

    def create(self, validated_data):
        # pop payment_method_id if not necessary for company creation
        validated_data.pop('payment_method_id', None)
        company = super().create(validated_data)

        # get user from request
        user = self.context.get("request").user
        # assign company to user
        user.company = company
        user.save()

        # assign user to be company's point_of_contact
        company.point_of_contact = user
        company.save()

        return company


class SubscriptionSerializer(serializers.ModelSerializer):
    price = serializers.SerializerMethodField()

    class Meta:
        model = Subscription
        fields = (
            'id',
            'status',
            'price',
        )

    def get_price(self, obj):
        if obj.plan:
            return obj.plan.amount_decimal / 100
        return None


class CompanyUpdateSerializer(serializers.ModelSerializer):
    payment_method_id = serializers.CharField(write_only=True)
    current_subscription = SubscriptionSerializer(read_only=True)

    class Meta:
        model = Company
        fields = (
            'id',
            'current_subscription',
            'name',
            'number_of_doors',
            'street_1',
            'street_2',
            'city',
            'state',
            'zip_code',
            'country',
            'assistant_phone_number',
            'payment_method_id',  # stripe payment method id
            'assistant_phone_is_verified',
        )

    def update(self, instance, validated_data):
        payment_method_id = validated_data.pop('payment_method_id', None)

        # call super update
        if validated_data:
            instance = super().update(instance, validated_data)

        if payment_method_id:
            # TODO Test this
            try:
                if not instance.customer_stripe_id:
                    # If no customer exists fails, create Stripe customer
                    stripe_customer = stripe.Customer.create(
                        payment_method=payment_method_id,
                        email=instance.users.first().email,
                        name=instance.name,
                        invoice_settings={
                            'default_payment_method': payment_method_id,
                        },
                    )
                    # Update the customer_stripe_id field with the Stripe customer's id
                    instance.customer_stripe_id = stripe_customer.id

            except stripe.error.StripeError as e:
                # Handle Stripe errors here
                raise serializers.ValidationError(f"Stripe error: {e}")

            instance.save()

            #  TODO Ask somebody if I need to do this
            # try:
            #     # Sync customer with dj-stripe
            #     customer, created = Customer.get_or_create(subscriber=instance.users.first())
            #     customer.api_retrieve()
            #
            #     # Avoid an unnecessary API call by using the data from stripe_customer
            #     customer.default_payment_method = PaymentMethod.sync_from_stripe_data(
            #         // set the default payment method here
            #     )
            #     customer.save()
            # except Exception as e:  # Replace with a more specific exception if possible
            #     # Handle other errors here
            #     raise serializers.ValidationError(f"Error updating customer: {e}")
            #
            #     # Save Stripe customer ID in Company model
            # instance.save()

        return instance
