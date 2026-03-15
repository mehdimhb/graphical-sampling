from graphical_sampling.structs import MaxHeap


def test_push():
    h = MaxHeap[int]()
    h.push(1)
    h.push(2)
    h.push(3)
    assert h.pop() == 3
    assert h.pop() == 2
    assert h.pop() == 1
    assert not h


def test_randompop():
    h = MaxHeap[int]()
    h.push(1)
    h.push(2)
    h.push(3)
    p1 = h.randompop()
    p2 = h.randompop()
    p3 = h.randompop()
    assert p1 != p2 != p3
    assert p1 in {1, 2, 3}
    assert p2 in {1, 2, 3} - {p1}
    assert p3 in {1, 2, 3} - {p1, p2}
    assert not h


def test_copy():
    h = MaxHeap[int]()
    h.push(1)
    h.push(2)
    h.push(3)
    h2 = h.copy()
    assert h == h2
    h.pop()
    assert h != h2
    h2.pop()
    assert h == h2
    h.pop()
    h2.pop()
    assert h == h2
    h.pop()
    h2.pop()
    assert h == h2
    assert not h
    assert not h2


def test_len():
    h = MaxHeap[int]()
    assert len(h) == 0
    h.push(1)
    assert len(h) == 1
    h.push(2)
    assert len(h) == 2
    h.push(3)
    assert len(h) == 3
    h.pop()
    assert len(h) == 2
    h.pop()
    assert len(h) == 1
    h.pop()
    assert len(h) == 0


def test_iter():
    h = MaxHeap[int]()
    h.push(1)
    h.push(2)
    h.push(3)
    assert set(h) == {1, 2, 3}
    assert len(h) == 3


def test_str():
    h = MaxHeap[int]()
    h.push(1)
    h.push(2)
    h.push(3)
    assert str(h) == "[3, 1, 2]"
