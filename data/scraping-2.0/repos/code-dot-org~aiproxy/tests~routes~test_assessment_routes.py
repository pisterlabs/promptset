import openai


class TestPostAssessment:
    """ Tests POST to '/assessment' to start an assessment.
    """

    def test_should_return_400_when_no_code(self, client, randomstring):
        response = client.post('/assessment', data={
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "examples": "[]",
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "1",
          "temperature": "0.2",
        })
        assert response.status_code == 400

    def test_should_return_400_when_no_prompt(self, client, randomstring):
        response = client.post('/assessment', data={
          "code": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "examples": "[]",
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "1",
          "temperature": "0.2",
        })
        assert response.status_code == 400

    def test_should_return_400_when_no_rubric(self, client, randomstring):
        response = client.post('/assessment', data={
          "code": randomstring(10),
          "prompt": randomstring(10),
          "api-key": randomstring(10),
          "examples": "[]",
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "1",
          "temperature": "0.2",
        })
        assert response.status_code == 400

    def test_should_return_400_on_openai_error(self, mocker, client, randomstring):
        mocker.patch('lib.assessment.assess.grade').side_effect = openai.error.InvalidRequestError('', '')
        response = client.post('/assessment', data={
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "examples": "[]",
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "1",
          "temperature": "0.2",
        })
        assert response.status_code == 400

    def test_should_return_400_when_passing_not_a_number_to_num_responses(self, client, randomstring):
        response = client.post('/assessment', data={
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "examples": "[]",
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "x",
          "temperature": "0.2",
        })
        assert response.status_code == 400

    def test_should_return_400_when_passing_not_a_number_to_temperature(self, client, randomstring):
        response = client.post('/assessment', data={
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "examples": "[]",
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "x",
        })
        assert response.status_code == 400

    def test_should_return_400_when_the_grade_function_does_not_return_data(self, mocker, client, randomstring):
        grade_mock = mocker.patch('lib.assessment.assess.grade')
        grade_mock.return_value = []

        response = client.post('/assessment', data={
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "examples": "[]",
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "0.2",
        })

        assert response.status_code == 400

    def test_should_return_400_when_the_grade_function_does_not_return_the_right_structure(self, mocker, client, randomstring):
        grade_mock = mocker.patch('lib.assessment.assess.grade')
        grade_mock.return_value = {
            'metadata': {},
            'data': {}
        }

        response = client.post('/assessment', data={
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "examples": "[]",
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "x",
        })

        assert response.status_code == 400

    def test_should_pass_arguments_to_grade_function(self, mocker, client, randomstring):
        grade_mock = mocker.patch('lib.assessment.assess.grade')
        data = {
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "examples": "[]",
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "0.2",
        }

        response = client.post('/assessment', data=data)

        grade_mock.assert_called_with(
            code=data["code"],
            prompt=data["prompt"],
            rubric=data["rubric"],
            examples=[],
            api_key=data["api-key"],
            llm_model=data["model"],
            remove_comments=True,
            num_responses=2,
            temperature=0.2
        )

    def test_should_return_the_result_from_grade_function_when_valid(self, mocker, client, randomstring):
        grade_mock = mocker.patch('lib.assessment.assess.grade')
        grade_mock.return_value = {
            'metadata': {},
            'data': [
                {
                    'Key Concept': randomstring(10),
                    'Observations': 'foo',
                    'Grade': 'No Evidence',
                    'Reason': 'bar'
                }
            ]
        }
        data = {
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "examples": "[]",
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "0.2",
        }

        response = client.post('/assessment', data=data)

        assert response.status_code == 200
        assert response.json == grade_mock.return_value


class TestPostTestAssessment:
    """ Tests POST to '/test/assessment' to start an assessment.
    """

    def test_should_return_400_on_openai_error(self, mocker, client, randomstring):
        mocker.patch('lib.assessment.assess.grade').side_effect = openai.error.InvalidRequestError('', '')
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        response = client.post('/test/assessment', data={
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "1",
          "temperature": "0.2",
        })
        assert response.status_code == 400

    def test_should_return_400_when_passing_not_a_number_to_num_responses(self, mocker, client, randomstring):
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        response = client.post('/test/assessment', data={
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "x",
          "temperature": "0.2",
        })
        assert response.status_code == 400

    def test_should_return_400_when_passing_not_a_number_to_temperature(self, mocker, client, randomstring):
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        response = client.post('/test/assessment', data={
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "x",
        })
        assert response.status_code == 400

    def test_should_return_400_when_the_grade_function_does_not_return_data(self, mocker, client, randomstring):
        grade_mock = mocker.patch('lib.assessment.assess.grade')
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        grade_mock.return_value = []

        response = client.post('/test/assessment', data={
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "0.2",
        })

        assert response.status_code == 400

    def test_should_return_400_when_the_grade_function_does_not_return_the_right_structure(self, mocker, client, randomstring):
        grade_mock = mocker.patch('lib.assessment.assess.grade')
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        grade_mock.return_value = {
            'metadata': {},
            'data': {}
        }

        response = client.post('/test/assessment', data={
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "x",
        })

        assert response.status_code == 400

    def test_should_pass_arguments_to_grade_function(self, mocker, client, randomstring):
        grade_mock = mocker.patch('lib.assessment.assess.grade')
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        data = {
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "0.2",
        }

        response = client.post('/test/assessment', data=data)

        grade_mock.assert_called_with(
            code='file data',
            prompt='file data',
            rubric='file data',
            api_key=data["api-key"],
            llm_model=data["model"],
            remove_comments=True,
            num_responses=2,
            temperature=0.2
        )

    def test_should_return_the_result_from_grade_function_when_valid(self, mocker, client, randomstring):
        grade_mock = mocker.patch('lib.assessment.assess.grade')
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        grade_mock.return_value = {
            'metadata': {},
            'data': [
                {
                    'Key Concept': randomstring(10),
                    'Observations': 'foo',
                    'Grade': 'No Evidence',
                    'Reason': 'bar'
                }
            ]
        }
        data = {
          "code": randomstring(10),
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "0.2",
        }

        response = client.post('/test/assessment', data=data)

        assert response.status_code == 200
        assert response.json == grade_mock.return_value


class TestPostBlankAssessment:
    """ Tests POST to '/test/assessment/blank' to start an assessment.
    """

    def test_should_return_400_on_openai_error(self, mocker, client, randomstring):
        mocker.patch('lib.assessment.assess.grade').side_effect = openai.error.InvalidRequestError('', '')
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        response = client.post('/test/assessment/blank', data={
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "1",
          "temperature": "0.2",
        })
        assert response.status_code == 400

    def test_should_return_400_when_passing_not_a_number_to_num_responses(self, mocker, client, randomstring):
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        response = client.post('/test/assessment/blank', data={
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "x",
          "temperature": "0.2",
        })
        assert response.status_code == 400

    def test_should_return_400_when_passing_not_a_number_to_temperature(self, mocker, client, randomstring):
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        response = client.post('/test/assessment/blank', data={
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "x",
        })
        assert response.status_code == 400

    def test_should_return_400_when_the_grade_function_does_not_return_data(self, mocker, client, randomstring):
        grade_mock = mocker.patch('lib.assessment.assess.grade')
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        grade_mock.return_value = []

        response = client.post('/test/assessment/blank', data={
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "0.2",
        })

        assert response.status_code == 400

    def test_should_return_400_when_the_grade_function_does_not_return_the_right_structure(self, mocker, client, randomstring):
        grade_mock = mocker.patch('lib.assessment.assess.grade')
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        grade_mock.return_value = {
            'metadata': {},
            'data': {}
        }

        response = client.post('/test/assessment/blank', data={
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "x",
        })

        assert response.status_code == 400

    def test_should_pass_arguments_including_blank_code_to_grade_function(self, mocker, client, randomstring):
        grade_mock = mocker.patch('lib.assessment.assess.grade')
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        data = {
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "0.2",
        }

        response = client.post('/test/assessment/blank', data=data)

        grade_mock.assert_called_with(
            code='',
            prompt='file data',
            rubric='file data',
            api_key=data["api-key"],
            llm_model=data["model"],
            remove_comments=True,
            num_responses=2,
            temperature=0.2
        )

    def test_should_return_the_result_from_grade_function_when_valid(self, mocker, client, randomstring):
        grade_mock = mocker.patch('lib.assessment.assess.grade')
        mock_open = mocker.mock_open(read_data='file data')
        mock_file = mocker.patch('builtins.open', mock_open)
        grade_mock.return_value = {
            'metadata': {},
            'data': [
                {
                    'Key Concept': randomstring(10),
                    'Observations': 'foo',
                    'Grade': 'No Evidence',
                    'Reason': 'bar'
                }
            ]
        }
        data = {
          "prompt": randomstring(10),
          "rubric": randomstring(10),
          "api-key": randomstring(10),
          "model": randomstring(10),
          "remove-comments": "1",
          "num-responses": "2",
          "temperature": "0.2",
        }

        response = client.post('/test/assessment/blank', data=data)

        assert response.status_code == 200
        assert response.json == grade_mock.return_value
