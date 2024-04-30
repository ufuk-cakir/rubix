from rubix.pipeline import linear_pipeline as lp
from rubix.utils import read_yaml
import pytest
from pathlib import Path
from copy import deepcopy


@pytest.fixture
def pipeline_fixture():
    cfg = read_yaml(Path(__file__).parent / "demo.yml")

    return cfg


@pytest.fixture
def pipeline_fixture_full():
    cfg = read_yaml(Path(__file__).parent / "demo.yml")

    ppl = lp.LinearTransformerPipeline(cfg)

    ppl.register_transformer(add)
    ppl.register_transformer(sub)
    ppl.register_transformer(mult)
    ppl.register_transformer(div)

    return ppl


def add(x, s: float):
    return x + s


def mult(x, m: float):
    return x * m


def div(x, d: float):
    return x / d


def sub(x, s: float):
    return x - s


def test_pipeline_construction(pipeline_fixture):
    cfg = pipeline_fixture

    pipeline = lp.LinearTransformerPipeline(cfg)

    assert pipeline.config == cfg
    assert pipeline._pipeline == []
    assert pipeline._names == []
    assert pipeline.transformers == {}
    assert pipeline.expression is None
    assert pipeline.compiled_expression is None


def test_register_transformer(pipeline_fixture):
    cfg = pipeline_fixture

    pipeline = lp.LinearTransformerPipeline(cfg)

    assert pipeline.transformers == {}

    pipeline.register_transformer(add)
    pipeline.register_transformer(sub)
    pipeline.register_transformer(mult)
    pipeline.register_transformer(div)

    assert pipeline.transformers == {"add": add, "sub": sub, "mult": mult, "div": div}

    with pytest.raises(
        ValueError, match="A transformer of this name is already present"
    ):
        pipeline.register_transformer(add)


def test_update_pipeline(pipeline_fixture_full):
    pipeline = pipeline_fixture_full

    pipeline.update_pipeline("A")

    assert pipeline._names == ["X", "Z"]

    pipeline.update_pipeline("Not there")

    assert pipeline._names == ["X", "Z"]


def test_build_pipeline(pipeline_fixture_full):
    pipeline = pipeline_fixture_full

    pipeline.build_pipeline()

    assert pipeline._names == ["C", "B", "A", "X", "Z"]
    assert list(pipeline.pipeline.keys()) == pipeline._names


def test_build_pipeline_broken(pipeline_fixture_full):
    pipeline = pipeline_fixture_full
    cfg = deepcopy(pipeline.config)

    del pipeline.config["Transformers"]["C"]["name"]

    with pytest.raises(
        ValueError,
        match="Each node of a pipeline must have a config node containing 'name'",
    ):
        pipeline.build_pipeline()

    pipeline.config = cfg

    pipeline.config["Transformers"]["X"]["depends_on"] = None

    with pytest.raises(ValueError, match="There can only be one starting point"):
        pipeline.build_pipeline()


# def test_get_jaxpr(pipeline_fixture):

# def test_compile_expression(pipeline_fixture):

# def test_compile_element(pipeline_fixture):


# def test_get_jaxpr_for_element(pipeline_fixture):

# def test_update_pipeline(pipeline_fixture):

# def test_build_pipeline(pipeline_fixture):

# def test_build_expression(pipeline_fixture):

# def test_apply(pipeline_fixture):
