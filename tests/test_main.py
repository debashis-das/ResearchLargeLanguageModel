from app.main import main

def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello" in captured.out