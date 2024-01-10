import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from gpt_plugin_package.openai_api_response_content import OpenAIAPIResponseContent

def test_code_block_with_ruby_code_block():
    body = 'test_spec.rb\n```ruby\nRSpec.describe "stuff" do\nend\n```'
    content = OpenAIAPIResponseContent(body)
    code = content.code_block()
    assert code == 'RSpec.describe "stuff" do\nend'

def test_code_block_with_ruby_code_block_2():
    body = "calculator_spec.rb\n```ruby\nrequire_relative 'calculator'\n\nRSpec.describe Calculator do\n  end\n```"
    content = OpenAIAPIResponseContent(body)
    code = content.code_block()
    assert code == "require_relative 'calculator'\n\nRSpec.describe Calculator do\n  end"

def test_code_block_with_python_code_block():
    body = 'test.py\n```python\ndef test_stuff():\n    assert True\n```'
    content = OpenAIAPIResponseContent(body)
    code = content.code_block()
    assert code == 'def test_stuff():\n    assert True'

def test_filename_extraction():
    body = 'test_spec.rb\n"rspec test_spec.rb"\n\n```ruby\nRSpec.describe "stuff" do\nend\n```'
    content = OpenAIAPIResponseContent(body)
    filename = content.filename()
    assert filename == 'test_spec.rb'

def test_sloppy_filename_extraction():
    body = 'filename: test_spec.rb\n```ruby\nRSpec.describe "stuff" do\nend\n```'
    content = OpenAIAPIResponseContent(body)
    filename = content.filename()
    assert filename == 'test_spec.rb'

def test_trig():
    body = "lib/trigonometry.rb\n```ruby\nclass Trigonometry\n  def self.sin(angle)\n  end\n\n  def self.cos(angle)\n  end\n\n  def self.tan(angle)\n  end\nend"
    content = OpenAIAPIResponseContent(body)
    filename = content.filename()
    assert filename == 'lib/trigonometry.rb'

def test_trig_code_block():
    body = "lib/trigonometry.rb\n```ruby\nclass Trigonometry\n  def self.sin(angle)\n  end\n\n  def self.cos(angle)\n  end\n\n  def self.tan(angle)\n  end\nend"
    content = OpenAIAPIResponseContent(body)
    assert content.code_block() is not None
