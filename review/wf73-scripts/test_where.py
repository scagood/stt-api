def test_where():
    from parakeet_service import routes
    print("ROUTES", routes.__file__, "neighbour" if "nxt" in open(routes.__file__).read() else "orig")
