import json
from csv import DictReader, Error as CsvError
from io import TextIOWrapper

import cohere
from cohere.responses.classify import Example

from django.contrib import messages
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Q
from django.shortcuts import redirect
from django.shortcuts import render
from django.views.decorators.http import require_POST, require_http_methods

from .forms import CSVUploadForm, ReviewSelectForm, ProductSelectForm
from .models import Product, Review


def product_list(request):
    selected_products = request.GET.getlist('product')

    query = Q()
    if selected_products:
        if "all" not in selected_products:
            query = Q(product__asin__in=selected_products)
        reviews = Review.objects.filter(query)
    else:
        reviews = Review.objects.none()

    paginator = Paginator(reviews, 20)
    page_number = request.GET.get('page')

    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    all_products = Product.objects.all()
    return render(request, 'product_list.html', {
        'page_obj': page_obj,
        'all_products': all_products,
        'selected_products': selected_products
    })


def upload_csv(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                summary, data_for_display = handle_uploaded_file(request.FILES['file'])
                return render(request, 'upload_summary.html',
                              {'form': form, 'summary': summary, 'data': data_for_display})
            except ValueError as e:
                messages.error(request, f"Error processing file: {e}")
                return redirect('upload_csv')
    else:
        form = CSVUploadForm()
    return render(request, 'upload_csv.html', {'form': form})


def handle_uploaded_file(f):
    file = f.temporary_file_path() if hasattr(f, 'temporary_file_path') else TextIOWrapper(f, encoding='utf-8')
    summary = {'total_records': 0, 'successful_records': 0, 'failed_records': 0, 'errors': []}
    data_for_display = []

    try:
        with (open(file, newline='') if isinstance(file, str) else file) as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                summary['total_records'] += 1
                try:
                    # Check if product exists, create if not
                    product, created = Product.objects.get_or_create(
                        asin=row['asin'],
                        defaults={'description': '', 'embedding': None}  # default values
                    )

                    # Create review associated with the product
                    Review.objects.create(
                        product=product,
                        overall=row.get('overall', -1),
                        vote=row.get('vote', 0.0) if row.get('vote', 0.0) != '' else 0,
                        verified=row.get('verified', False),
                        review_text=row.get('reviewText', 'No review'),
                        summary_text=row.get('summary', 'No title'),
                    )

                    summary['successful_records'] += 1
                    data_for_display.append(row)
                except Exception as e:
                    summary['failed_records'] += 1
                    summary['errors'].append(str(e))

    except (CsvError, KeyError, ValueError) as e:
        raise ValueError(f"File processing error: {e}")

    return summary, data_for_display


def load_examples_from_json(file_path):
    with open(file_path, 'r') as file:
        examples_json = json.load(file)
    return [Example(e["text"], e["label"]) for e in examples_json]


def classify_reviews(reviews, examples_file_path):
    # Load your API key from a secure place, avoid hardcoding
    cohere_api_key = 'c4G53MUnZPuCr7reACDrUP5o20dvuTgvuO3QFpUP'
    co = cohere.Client(cohere_api_key)

    # Load examples
    examples = load_examples_from_json(examples_file_path)

    # Classify the reviews
    response = co.classify(
        model='embed-english-v2.0',
        inputs=reviews,
        examples=examples
    )
    # Process and return the response
    return [f'{x.prediction}: {x.labels[x.prediction][0] * 100:4.2f}' for x in response.classifications]


@require_http_methods(["GET", "POST"])
def review_classification_view(request):
    product_form = ProductSelectForm()
    review_form = None
    selected_product = None
    reviews_page = None

    if request.method == 'POST':
        selected_product_id = request.session.get('selected_product_id')
        if selected_product_id:
            selected_product = Product.objects.get(id=selected_product_id)

        if 'select_product' in request.POST:
            product_form = ProductSelectForm(request.POST)
            if product_form.is_valid():
                selected_product = product_form.cleaned_data['product']
                request.session['selected_product_id'] = selected_product.id
                reviews = Review.objects.filter(product=selected_product)
                paginator = Paginator(reviews, 10)  # 10 reviews per page
                reviews_page = paginator.get_page(1)
                review_form = ReviewSelectForm(initial={'reviews': reviews_page.object_list})

        elif 'classify' in request.POST:
            review_form = ReviewSelectForm(request.POST)
            if review_form.is_valid():
                if 'select_all' in request.POST:
                    selected_reviews = Review.objects.filter(product=selected_product)
                else:
                    selected_reviews = review_form.cleaned_data['reviews']

                reviews_to_classify = [review.get_review() for review in selected_reviews]
                classified_results = classify_reviews(reviews_to_classify,
                                                      'C:\\Users\\kaust\\PycharmProjects\\HonestReviewer\\HonestReviewer\\reviews\\templates\\classification_examples.json')

                for review, classification in zip(selected_reviews, classified_results):
                    review.helpfulness_classification = classification
                    review.save()

                messages.success(request, "Reviews classified successfully.")

                # Re-fetch the reviews to update the page with classifications
                reviews = Review.objects.filter(product=selected_product)
                paginator = Paginator(reviews, 10)
                reviews_page = paginator.get_page(1)
                review_form = ReviewSelectForm(initial={'reviews': reviews_page.object_list})


    elif request.method == 'GET':
        selected_product_id = request.session.get('selected_product_id')  # Retrieve product from session
        if selected_product_id:
            selected_product = Product.objects.get(id=selected_product_id)  # Get the selected product
            page = request.GET.get('page')
            reviews = Review.objects.filter(product=selected_product)
            paginator = Paginator(reviews, 10)  # 10 reviews per page
            reviews_page = paginator.get_page(page)

    return render(request, 'review_classification.html', {
        'product_form': product_form,
        'review_form': review_form,
        'selected_product': selected_product,
        'reviews_page': reviews_page
    })
