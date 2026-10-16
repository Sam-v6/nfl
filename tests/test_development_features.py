import torch


def test_motion_features_ignore_masked_frames_and_player_order() -> None:
	from development_features import motion_summary

	x = torch.zeros(2, 3, 22, 5)
	x[:, :, :11, 4] = 1
	x[:, -1, :, 0] = torch.arange(22)
	mask = torch.tensor([[True, False, True], [True, False, True]])
	before = motion_summary(x, mask)
	x[:, 1] = 9999
	torch.testing.assert_close(before, motion_summary(x, mask))
	permutation = torch.randperm(22)
	torch.testing.assert_close(before, motion_summary(x[:, :, permutation], mask))
	x[:, 0, :, 0] = x[:, -1, :, 0]
	assert not torch.equal(before, motion_summary(x, mask))
