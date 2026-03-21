import json
import pytest
from pydantic import BaseModel, Field
from app.core.compiler import FSMCompiler, JSONSchemaToRegexCompiler
from app.core.masker import TokenLogitMasker, default_vocab
from app.core.cache import FSMCache


class SimpleUser(BaseModel):
    id: int
    name: str
    is_active: bool


class NestedOrder(BaseModel):
    order_id: str
    total_amount: float
    status: str = Field(..., description="Order status")


@pytest.fixture
def compiler():
    return FSMCompiler()


def test_primitive_string_schema_compilation(compiler):
    schema = {"type": "string"}
    fsm = compiler.compile_schema_to_fsm(schema)
    assert fsm.accepts('"hello world"')
    assert fsm.accepts('""')
    assert not fsm.accepts('hello')


def test_primitive_integer_schema_compilation(compiler):
    schema = {"type": "integer"}
    fsm = compiler.compile_schema_to_fsm(schema)
    assert fsm.accepts('42')
    assert fsm.accepts('-100')
    assert fsm.accepts('0')
    assert not fsm.accepts('3.14')
    assert not fsm.accepts('"42"')


def test_enum_schema_compilation(compiler):
    schema = {
        "type": "string",
        "enum": ["USD", "EUR", "GBP", "JPY"],
    }
    fsm = compiler.compile_schema_to_fsm(schema)
    assert fsm.accepts('"USD"')
    assert fsm.accepts('"EUR"')
    assert not fsm.accepts('"CAD"')
    assert not fsm.accepts('USD')


def test_nested_object_schema_compilation(compiler):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    fsm = compiler.compile_schema_to_fsm(schema)
    valid_json = '{"name": "Alice", "age": 30}'
    assert fsm.accepts(valid_json)
    assert not fsm.accepts('{"name": "Alice"}')
    assert not fsm.accepts('{"age": 30}')


def test_pydantic_model_compilation(compiler):
    fsm = compiler.compile_schema_to_fsm(SimpleUser)
    valid_json = '{"id": 101, "name": "Bob", "is_active": true}'
    assert fsm.accepts(valid_json)
    assert not fsm.accepts('{"id": "one", "name": "Bob", "is_active": true}')


def test_token_logit_masker_initial_tokens():
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["SUCCESS", "FAILED"]},
        },
        "required": ["status"],
    }
    compiler = FSMCompiler()
    fsm = compiler.compile_schema_to_fsm(schema)
    masker = TokenLogitMasker(fsm=fsm, vocab=default_vocab)

    valid_token_ids = masker.get_valid_token_ids()
    valid_tokens_decoded = [default_vocab.decode([tid]) for tid in valid_token_ids if tid < 100256]

    assert any("{" in tok for tok in valid_tokens_decoded)
    assert not any(tok.strip() == "SUCCESS" for tok in valid_tokens_decoded)


def test_fsm_cache_retrieval():
    cache = FSMCache(capacity=5)
    schema = {"type": "integer"}

    fsm1 = cache.get_or_compile(schema)
    assert cache.size == 1

    fsm2 = cache.get_or_compile(schema)
    assert cache.size == 1
    assert fsm1 is fsm2
